import math
import random
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from utils import cluster 
from fast_cluster import compute_hashes, clustered_aggregate, clustered_broadcast

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()

        self.num_heads = args.num_heads
        self.hidden_units = args.item_hidden_units + args.user_hidden_units
        self.attention_head_size = int(self.hidden_units/args.num_heads)
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
        self.attn_dropout = nn.Dropout(args.dropout_rate)
        self.output_dropout = nn.Dropout(args.dropout_rate)

        self.dense = nn.Linear(self.hidden_units, self.hidden_units)
        self.layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
        
    def transpose_for_scores(self,x): #not currently used due to concat of user, item embedding
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)
    
    def forward(self, seq, attention_mask):

        
        mix_query = self.query(seq)
        mix_key= self.key(seq)
        mix_value = self.value(seq)

        query = self.transpose_for_scores(mix_query)
        key = self.transpose_for_scores(mix_key)
        value = self.transpose_for_scores(mix_value)
        
        attention_score = torch.matmul(query,key.transpose(-1,-2)) / self.sqrt_scale
        

        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).expand(attention_score.size(0), -1, -1)
            attention_score = attention_score + (attention_mask.unsqueeze(1).to(torch.float32) - 1) * 100000000
        else:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_score.size(1), -1, -1)
            attention_score = attention_score + (attention_mask.to(torch.float32) - 1) * 100000000

        attention_prob = F.softmax(attention_score, dim=-1)
        #attention_prob = self.softmax(attention_score)
        attention_prob = self.attn_dropout(attention_prob)

        context = torch.matmul(attention_prob, value)
        
        # not needed without using MHA
        context = context.permute(0,2,1,3).contiguous() # check how shape is computed
        new_context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_layer_shape)


        hidden_state = self.dense(context)
        hidden_state = self.output_dropout(hidden_state)
        hidden_state = self.layernorm(hidden_state + seq) #residual connection

        return hidden_state # return attention map if needed


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
        
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)
        import IPython; IPython.embed(colors='Linux'); exit(1)
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
    def __init__(self, args, iterations =1, bits =32):
        super(Clustered_Attention, self).__init__()

        self.args = args
        self.num_heads = args.num_heads
        self.hidden_units = args.item_hidden_units + args.user_hidden_units
        self.attention_head_size = int(self.hidden_units / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        
        self.clusters = args.cluster_num
        self.bits = bits
        self.iterations =iterations
        self.dropout = nn.Dropout(args.dropout_rate)
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

        # sorted_clusters: N, H, L
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _group_queries(self, Q, groups, lengths):
        """
        Aggregate the Qs based on the index of cluster they belong to. Make
        sure to allow for gradient propagation backwards from the grouped
        queries to each query.
        """

        q_grouped = _GroupQueries.apply(Q, *groups, lengths) # 128 10 50
        
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

        # initalize query_length to match the number of queries being processed
        query_lengths = torch.full((N * H * L,), L, dtype=torch.int64) # check for validity

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
        Q_grouped = Q_grouped_.view(N,H,-1,E)

        
        # Compute attention

        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)
        
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1)) # N H L E
        V = torch.einsum("nhls,nhsd->nhld", A, values) # N H L E 

        # Broadcast grouped attention


        V_broadcast = self._broadcast_values(V, groups, query_lengths)
        


        



class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward,self).__init__()

        self.hidden_units = args.item_hidden_units + args.user_hidden_units
        self.inner_layer = nn.Linear(self.hidden_units,self.hidden_units)
        self.activation = nn.GELU()
        self.outer_layer = nn.Linear(self.hidden_units,self.hidden_units)
        self.layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self,seq):
        hidden_state = self.layernorm(hidden_state)

        hidden_state = self.inner_layer(seq)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.outer_layer(hidden_state)

        hidden_state = self.dropout(hidden_state)
        

        return hidden_state


        

class EncoderLayer(nn.Module):
    def __init__ (self, args):
        super(EncoderLayer, self).__init__()

        self.base_attention = SelfAttention(args)
        self.cluster_attention = Clustered_Attention(args)
        self.feedforward = FeedForward(args)

    def forward(self, hidden_state, attention_mask, args):
        if args.attention == 'base':
            attention_output = self.base_attention(hidden_state, attention_mask)
        elif args.attention == 'cluster':
            attention_output = self.cluster_attention(hidden_state, attention_mask)

            
        feedforward_output = self.feedforward(attention_output)

        return feedforward_output



class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()

        layer = EncoderLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_blocks)])
    
    def forward(self, hidden_state, attention_mask, timeline_mask ,args):
        all_encoder_layer = []

        for layer_module in self.layer:
            hidden_state = layer_module(hidden_state, attention_mask,args)
            hidden_state *= ~timeline_mask.unsqueeze(-1)
            all_encoder_layer.append(hidden_state)

        return all_encoder_layer
