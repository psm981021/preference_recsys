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

        # check validity
        # batch_sample_one = F.normalize(batch_sample_one, p=2, dim=1)
        # batch_sample_two = F.normalize(batch_sample_two, p=2, dim=1)

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
            # if sim22.shape != torch.Size([1]):
            #     sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1) # positive
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1) # negative
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        

        nce_loss = self.criterion(logits, labels)
        return nce_loss
    
class AlignmentLossWithSinkhorn(nn.Module):
    def __init__(self, sinkhorn_iterations=20, epsilon=0.05):
        super(AlignmentLossWithSinkhorn, self).__init__()
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

    def compute_similarity(self, cl_seq2intents, seq2intents):
        # Dot product and exponential scaling
        
        scores = torch.matmul(cl_seq2intents, seq2intents.t())  # [K, K] if K is the number of prototypes/intents
        scores = torch.exp(scores / self.epsilon)  # Apply exponential scaling
        return scores

    def distributed_sinkhorn(self, out):
        Q = out.t()  # Transpose for consistency (prototypes x samples)
        B = Q.shape[1]  # number of samples (from seq2intents)
        K = Q.shape[0]  # number of prototypes (from cl_seq2intents)

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def compute_alignment_loss(self, cl_seq2intents, seq2intents, assignment_matrix):
        aligned_intents = torch.matmul(assignment_matrix, seq2intents)  # Re-order seq2intents as per assignments
        loss = F.mse_loss(aligned_intents, cl_seq2intents)
        return loss
    
    def forward(self, cl_seq2intents, seq2intents):
        # Compute similarity and apply Sinkhorn normalization
        similarity_scores = self.compute_similarity(cl_seq2intents, seq2intents)
        assignment_matrix = self.distributed_sinkhorn(similarity_scores)
        loss = self.compute_alignment_loss(cl_seq2intents, seq2intents, assignment_matrix)
        return loss

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
        attention_probs =[]
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
            # attention_probs.append(attention_prob)

        outputs = torch.cat(attention_outputs, dim=0)
        # attention_prob = torch.cat(attention_probs, dim=0)
        

        # concat after attention
        reverse_indices = torch.argsort(sorted_indices)
        seq_sort = outputs.view(N,-1)
        output = seq_sort[reverse_indices].view(N,C,E)
        
        # sorted_attention_map = attention_prob[reverse_indices]
        return output
        
class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate,self).__init__()

        self.hidden_units = args.hidden_size# + args.user_hidden_units
        self.inner_layer = nn.Linear(self.hidden_units,self.hidden_units*4)
        self.activation = nn.GELU()
        self.outer_layer = nn.Linear(self.hidden_units*4,self.hidden_units)
        self.layernorm = LayerNorm(self.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self,seq):
        
        hidden_state = self.inner_layer(seq)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.outer_layer(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layernorm(hidden_state+seq)

        return hidden_state

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class Layer(nn.Module):
    def __init__ (self, args):
        super(Layer, self).__init__()

        self.base_attention = SelfAttention(args)
        #self.cluster_attention = Clustered_Attention(args)
        self.cluster_attention_chunking = Clustered_Attention_Chunking(args)
        self.feedforward = Intermediate(args)

    def forward(self, hidden_state, attention_mask,args,cluster_id=None):
        if cluster_id is not None:
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
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask,args,cluster_id, output_all_encoded_layers=True):

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask,args, cluster_id)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers