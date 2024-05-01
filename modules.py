import math
import random
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
from fast_cluster import compute_hashes, clustered_aggregate, clustered_broadcast
import matplotlib.pyplot as plt
import seaborn as sns
import random

class _GroupQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, clusters, counts, lengths):
        factors = 1./counts.float()
        q_grouped = clustered_aggregate(Q, clusters, factors, lengths)
        ctx.save_for_backward(clusters, counts, factors)

        return q_grouped

    @staticmethod
    def backward(ctx, grad_q_grouped):
        clusters, counts, factors = ctx.saved_tensors
        grad_q = clustered_broadcast(grad_q_grouped, clusters, counts, factors)

        return grad_q, None, None, None

class _BroadcastValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_grouped, clusters, counts, lengths):
        
        """
        factors N H C
        v_grouped N H C E
        counts N H C
        """
        # N x H x C
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)

        # N x H x L x E
        V = clustered_broadcast(v_grouped, clusters, counts, factors)
        ctx.save_for_backward(clusters, counts, factors, lengths)

        return V
    @staticmethod
    def backward(ctx, grad_v):
        clusters, counts, factors, lengths = ctx.saved_tensors
        grad_v_grouped = clustered_aggregate(grad_v, clusters, factors, lengths)

        return grad_v_grouped, None, None, None

class Clustered_Attention(nn.Module):
    """
    original code from https://github.com/idiap/fast-transformers/tree/master

    Use LSH and clustering in Hamming space to group query - minimal L2 distance

    Given Q, K, V cluster the Q into groups in "C" and compute the "C" query centroids Q_c

    We now use to the centroids Q_c to compute the attention using:

        V'_c = softmax(Q_c.mm(K.t()), dim=-1).mm(V).

    Now the computed values V'_c are "broadcasted" back to the query members
    of the corresponding cluster.

    """
    #iterations change to 10
    def __init__(self, args, iterations =5, bits =32):
        super(Clustered_Attention, self).__init__()

        self.args = args
        self.num_heads = args.num_hidden_layers
        self.hidden_units = args.hidden_size
        self.attention_head_size = int(self.hidden_units / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        
        self.clusters = args.num_intent_clusters
        self.bits = bits
        self.iterations =iterations
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        self.query = nn.Linear(self.hidden_units, self.all_head_size)
        self.key = nn.Linear(self.hidden_units, self.all_head_size)
        self.value = nn.Linear(self.hidden_units, self.all_head_size)
    
    def transpose_for_scores(self,x): #not currently used due to concat of user, item embedding
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)
    
    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape


        planes = Q.new_empty((self.bits, E+1)) # assign number of hashes for representation 
        torch.nn.init.normal_(planes) #initialize with normal distributuon

        planes[:,-1] = 0
        
        Q = Q.contiguous() #in order to use view, since stride is not working properly
        Q_reshaped = Q.view(N*H*L, E)
       

        hashes_ = compute_hashes(Q_reshaped, planes) # [N*H*L] shape
        hashes = hashes_.view(N,H,L)

        clusters, counts =  cluster(
            hashes,
            query_lengths,
            self.args,
            iterations=self.iterations,
            bits=self.bits
        ) 
        # clusters N H L
        # counts N H C 

        # sorted_clusters, sorted_indx: N, H, L
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _group_queries(self, Q, groups, lengths):
        """
        Aggregate the Qs based on the index of cluster they belong to. Make
        sure to allow for gradient propagation backwards from the grouped
        queries to each query.
        """

        q_grouped = _GroupQueries.apply(Q, *groups, lengths) 
        
        return q_grouped
    
    def _broadcast_values(self, V, groups, lengths):
        """
        Broadcast the values back to the correct positions but make sure
        that the gradient flows properly.

        """
        V_new = _BroadcastValues.apply(V.contiguous(), *groups, lengths)
        return V_new


    def forward(self, seq, attn_mask):
        
        # cluster attention does not use attention mask
        
        mix_query = self.query(seq)
        mix_key= self.key(seq)
        mix_value = self.value(seq)

        queries = self.transpose_for_scores(mix_query)
        keys = self.transpose_for_scores(mix_key)
        values = self.transpose_for_scores(mix_value)

        
        N, H, L, E = queries.shape
        _, _, S, D = values.shape

        softmax_temp = 1./math.sqrt(E)

        # initalize query_length to match the query length
        query_lengths = torch.full((N * H * L,), L, dtype=torch.int64)

        # used as cluster lengths, sequence length for each sequence in hashes
        key_lengths = torch.full((N , H , L), L, dtype=torch.int64).unsqueeze(2).to(seq.device)

        # cluster the queries into groups
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)

        # Re-organize queries so that first group belong to first cluster
        # next to second cluster and so on. This improves kernel implementations
        
        q_offset = torch.arange(N*H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N*H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N,H,L,E)


        # Aggregate the re-arranged queries
        
        Q_grouped_ = self._group_queries(s_queries, groups, query_lengths)
        Q_grouped = Q_grouped_.view(N,H,-1,E) # N H C E 

        
        # Compute attention

        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)
        
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1)) # N H C E
        V = torch.einsum("nhls,nhsd->nhld", A, values) # N H C E 

        # Broadcast grouped attention
        V_broadcast = self._broadcast_values(V, groups, query_lengths) # N H L E
        
        # Reverse the privious mapping

        rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (rev_indx.view(N*H, -1) + q_offset).reshape(-1)
        V_new = V_broadcast.reshape(-1, D).index_select(0, q_rev_flat).view(N,H,L,D)
        V_new = V_new.permute(0, 2, 1, 3).contiguous().view(N,L,-1) # N L H C
        
        # add normalization , dropout residual connection if needed
        return V_new
    


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature, device, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.total_calls = 0
        self.call_with_repeat_seq = 0

    def forward(self, features, intents=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # check probability of intent belongs to the same intent
        if intents is not None:
            unique_intents = torch.unique(intents)
            if unique_intents.shape[0] != intents.shape[0]:
                self.call_with_repeat_seq += 1
            self.total_calls += 1
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # normalize features
        features = F.normalize(features, dim=2)

        batch_size = features.shape[0]
        if intents is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif intents is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif intents is not None:
            intents = intents.contiguous().view(-1, 1)
            if intents.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(intents, intents.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        #         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class PCLoss(nn.Module):
    """ Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168
    """

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features: 
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_id)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_id)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        # don't do de-noise
        else:
            for intent in intents:
                
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss

class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        # sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        # sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        # sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            #intent_ids should be list

            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")

        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            if sim22.shape != torch.Size([1]):
                sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1) # positive
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1) # negative
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    code: https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py
    """

    LARGE_NUMBER = 1e9

    def __init__(self, tau=1.0, gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.0

    def forward(self, batch_sample_one, batch_sample_two):
        z = torch.cat([batch_sample_one, batch_sample_two], dim=0)
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm
        return loss
    
def plot_attention_map(attention_map,args):
    x_labels = [f'{index}' if index % 5 == 0 else '' for index in range(50)]
    y_labels = [f'{index}' if index % 5 == 0 else '' for index in range(50)]

    plt.figure(figsize=(15, 15))
    sns.heatmap(attention_map, cmap='viridis', annot=False, xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Item Index')
    plt.ylabel('Item Index')
    plt.title('Attention Map')
    
    # Save the figure with a random index, model_idx seems to be missing, using a placeholder
    plt.savefig(f'attention_map_{random.randint(0, 100)}.png')
    plt.show()

class Clustered_Attention_Chunking(nn.Module):
    def __init__(self, args):
        super(Clustered_Attention_Chunking, self).__init__()
        self.args =args
        self.attention = SelfAttention(args)
        pass

    def forward(self, seq, attention_mask, cluster_id = None):
        #chunking
        N,C,E = seq.shape
        chunk_size = N // int(self.args.num_intent_clusters)
        N = chunk_size * int(self.args.num_intent_clusters)

        seq_sort = seq.view(N, -1)
        if isinstance(cluster_id, list):
            if N == len(cluster_id[0]):
                sorted_indices = torch.argsort(cluster_id[0]) #cluster_id input as list
            else:
                cluster_id = torch.cat((cluster_id[0], cluster_id[0]), dim=0)
                sorted_indices = torch.argsort(cluster_id)
        else:
            cluster_id = torch.cat((cluster_id,cluster_id),dim=0)
            sorted_indices = torch.argsort(cluster_id)

        try:    
            seq_sorted = seq_sort[sorted_indices].view(N,C,E)
        except:
            print("\nClustered Attention Debugging");import IPython; IPython.embed(colors='Linux');exit(1);
        
        attention_outputs = []
        for i in range(int(self.args.num_intent_clusters)):
            #use chunking
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)

            key_start_idx = max((i - 1) * chunk_size, 0)
            key_end_idx = min(((i + 1) * chunk_size if i > 1 else 2*chunk_size), N)
        
            query_chunk_seq = seq_sorted[start_idx:end_idx, :, :]
            key_chunk_seq = seq_sorted[key_start_idx:key_end_idx, :, :]

            chunk_seq = seq_sorted[start_idx:end_idx,:, :]
            chunk_attention_mask_ = attention_mask[start_idx:end_idx,: , :, :]

            
            if self.args.vanilla_attention == True:
                attention_output = self.attention(query_chunk_seq,chunk_attention_mask_,key_chunk_seq)
            else:
                attention_output = self.attention(chunk_seq,chunk_attention_mask_)
            
            attention_outputs.append(attention_output)

        outputs = torch.cat(attention_outputs, dim=0)

        # concat after attention
        reverse_indices = torch.argsort(sorted_indices)
        seq_sort = outputs.view(N,-1)
        output = seq_sort[reverse_indices].view(N,C,E)
        
        # start_idx = torch.arange(0, N, chunk_size)
        # end_idx = start_idx + chunk_size
        # chunk_seq = seq_sorted[start_idx[:, None], end_idx[:, None], :].reshape(-1, C, E)

        # chunk_attention_mask_ = attention_mask[start_idx[:, None], :, :, :]
        # chunk_attention_mask_ = chunk_attention_mask_.reshape(-1, chunk_attention_mask_.shape[2], chunk_attention_mask_.shape[3])

        # attention_output = self.attention(chunk_seq, chunk_attention_mask_)
        # outputs = attention_output.reshape(self.args.num_intent_clusters, chunk_size, C, E)

        # reverse_indices = torch.argsort(sorted_indices)
        # seq_sort = outputs.view(N, -1)
        # output = seq_sort[reverse_indices].view(N, C, E)
    

        return output
        

        

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()

        self.num_heads = args.num_hidden_layers
        self.hidden_units = args.hidden_size #+ args.user_hidden_units
        self.attention_head_size = int(self.hidden_units/args.num_hidden_layers)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.sqrt_scale = math.sqrt(self.hidden_units)

        self.query = nn.Linear(self.hidden_units, self.all_head_size)
        self.key = nn.Linear(self.hidden_units, self.all_head_size)
        self.value = nn.Linear(self.hidden_units, self.all_head_size)

        # reuse when MHA is not necessary
        # self.query = nn.Linear(self.hidden_units, self.hidden_units)
        # self.key = nn.Linear(self.hidden_units, self.hidden_units)
        # self.value = nn.Linear(self.hidden_units, self.hidden_units)
    
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.dense = nn.Linear(self.hidden_units, self.hidden_units)
        self.layernorm = nn.LayerNorm(self.hidden_units, eps=1e-12)
        
    def transpose_for_scores(self,x): #not currently used due to concat of user, item embedding
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)
    
    def forward(self, seq, attention_mask,key=None):
        if key is not None:

            mix_query = self.query(seq)
            mix_key = self.key(key)
            mix_value = self.value(key)

            query = self.transpose_for_scores(mix_query)
            key = self.transpose_for_scores(mix_key)
            value = self.transpose_for_scores(mix_value)

            b, h, l, _ = key.shape
            key_slice = torch.split(key, b // 2)
            value_slice = torch.split(value, b // 2)
                                           
            attention_score1 = torch.matmul(query,key_slice[0].transpose(-1,-2)) / self.sqrt_scale 
            attention_score2 = torch.matmul(query,key_slice[1].transpose(-1,-2)) / self.sqrt_scale

            attention_score1 = attention_score1 + attention_mask
            attention_prob_1 = nn.Softmax( dim=-1)(attention_score1)
            attention_prob_1 = self.attn_dropout(attention_prob_1)

            attention_score2 = attention_score2 + attention_mask
            attention_prob_2 = nn.Softmax( dim=-1)(attention_score2)
            attention_prob_2 = self.attn_dropout(attention_prob_2)

            context_1 = torch.matmul(attention_prob_1, value_slice[0])
            context_2 = torch.matmul(attention_prob_2, value_slice[1])

            context = (context_1 + context_2) / 2.0
            context = context.permute(0,2,1,3).contiguous() 

        else:
            mix_query = self.query(seq)
            mix_key = self.key(seq)
            mix_value = self.value(seq)

            query = self.transpose_for_scores(mix_query)
            key = self.transpose_for_scores(mix_key)
            value = self.transpose_for_scores(mix_value)

            attention_score = torch.matmul(query,key.transpose(-1,-2)) / self.sqrt_scale

            attention_score = attention_score + attention_mask

            attention_prob = nn.Softmax( dim=-1)(attention_score)
            attention_prob = self.attn_dropout(attention_prob)

            context = torch.matmul(attention_prob, value)
            context = context.permute(0,2,1,3).contiguous() 

        new_context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_layer_shape)

        hidden_state = self.dense(context)
        hidden_state = self.output_dropout(hidden_state)
        hidden_state = self.layernorm(hidden_state + seq)

        return hidden_state



class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward,self).__init__()

        self.hidden_units = args.hidden_size# + args.user_hidden_units
        self.inner_layer = nn.Linear(self.hidden_units,self.hidden_units*4)
        self.activation = nn.GELU()
        self.outer_layer = nn.Linear(self.hidden_units*4,self.hidden_units)
        self.layernorm = nn.LayerNorm(self.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self,seq):
        
        hidden_state = self.inner_layer(seq)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.outer_layer(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layernorm(hidden_state+seq)

        return hidden_state


        

class EncoderLayer(nn.Module):
    def __init__ (self, args):
        super(EncoderLayer, self).__init__()

        self.base_attention = SelfAttention(args)
        #self.cluster_attention = Clustered_Attention(args)
        self.cluster_attention_chunking = Clustered_Attention_Chunking(args)
        self.feedforward = FeedForward(args)

    def forward(self, hidden_state, attention_mask,args,cluster_id):
        if cluster_id is not None:
        # if args.attention_type == "Cluster" and hasattr(args, 'cluster_id'):
            # perform Clustered Attention
            attention_output = self.cluster_attention_chunking(hidden_state, attention_mask,cluster_id)

        else:
            # perform Self-Attention
            attention_output = self.base_attention(hidden_state, attention_mask) 

        feedforward_output = self.feedforward(attention_output)

        return feedforward_output



class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.args =args
        layer = EncoderLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])
    
    def forward(self, hidden_state, attention_mask,args,cluster_id):
        all_encoder_layer = []

        for layer_module in self.layer:
            hidden_state = layer_module(hidden_state, attention_mask,args,cluster_id)
            all_encoder_layer.append(hidden_state)

        return all_encoder_layer
