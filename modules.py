import math
import random
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from attention import *
from attention.clustered_attention import ClusteredAttention

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
        #attention_score = attention_score + (attention_mask.unsqueeze(1).to(torch.float32) - 1) * 100000000 #modify for fit
            attention_mask = attention_mask.unsqueeze(0).expand(attention_score.size(0), -1, -1)
            attention_score = attention_score + (attention_mask.unsqueeze(1).to(torch.float32) - 1)
        else:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_score.size(1), -1, -1)
            attention_score = attention_score + (attention_mask.to(torch.float32) - 1)
            

        
        attention_prob = self.softmax(attention_score)
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

        hidden_state = self.inner_layer(seq)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.outer_layer(hidden_state)

        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layernorm(hidden_state)

        return hidden_state


        

class Layer(nn.Module):
    def __init__ (self, args):
        super(Layer, self).__init__()

        self.base_attention = SelfAttention(args)
        self.cluster_attention = ClusteredAttention(args)
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

        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_blocks)])
    
    def forward(self, hidden_state, attention_mask, timeline_mask ,args):
        all_encoder_layer = []

        for layer_module in self.layer:
            hidden_state = layer_module(hidden_state, attention_mask,args)
            hidden_state *= ~timeline_mask.unsqueeze(-1)
            all_encoder_layer.append(hidden_state)

        return all_encoder_layer
