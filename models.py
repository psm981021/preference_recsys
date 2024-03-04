import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import plot_blobs
from kmeans_torch import kmeans, kmeans_predict
from sklearn.decomposition import PCA
from modules import Encoder

class UPTRec(torch.nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(UPTRec, self).__init__()

        self.itemnum = itemnum
        self.usernum = usernum
        self.dev = args.device
        self.hidden_units = args.item_hidden_units + args.user_hidden_units
        
        self.item_embedding = nn.Embedding(self.itemnum+1,args.item_hidden_units)
        self.user_embedding = nn.Embedding(usernum+1,args.user_hidden_units) 
        self.position_embedding = nn.Embedding(args.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

        self.encoder = Encoder(args)

        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self,user_ids, seq, pos_seqs, neg_seqs):
        '''
        B : Batch
        T : seq length
        C : hidden_unit
        '''
        #--- embedding --- 

        #item embedding 
        seq_ = self.item_embedding(torch.LongTensor(seq).to(self.dev))

        #positional encoding
        positions = torch.arange(seq_.size(1)).unsqueeze(0).expand(seq_.size(0),-1)

        #position embedding
        positions = self.position_embedding(positions.to(self.dev))

        #user embedding  B x C to B x T x C
        u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).unsqueeze(1).repeat(1,seq_.size(1),1) 

        # concat user embedding with item embedding
        seq_ = torch.cat([seq_,u_latent], dim =2).view(seq_.size(0),-1,self.hidden_units )
        seq_ += positions
        
        #dropout
        seq_ = self.emb_dropout(seq_)
        
        
        
        #---kmeans pytorch module--- change to class if needed

        # In batch
        batch_cluster_ids =[]
        for batch in range(seq_.size(0)):
            seq_cluster = seq_[batch]
            
            seq_cluster_id, cluster_centers = kmeans(
                X=seq_cluster,
                num_clusters= 10, 
                distance = 'euclidean', 
                tqdm_flag=False,
                #iter_limit = 20,
                device = self.dev
                
                
            )
            seq_cluster_id = seq_cluster_id.to(self.dev)

            # check whether values have been change every batch
            cluster_centers = cluster_centers.to(self.dev) 

            batch_cluster_ids.append(seq_cluster_id.view(-1,1))

        #timeline masking
        
        timeline_mask = torch.BoolTensor(seq == 0).to(self.dev)
        seq_ *= ~timeline_mask.unsqueeze(-1) # Brodacast in last dim

        # --- cluster mask --- 
        t1 = seq_.shape[1] #T

        #for now use torch.tril for test
        attention_mask = ~torch.tril(torch.ones((t1,t1), dtype=torch.bool, device= self.dev))


        # --- attention layer ---
        logits = self.encoder(seq_ ,attention_mask, timeline_mask) # logits contains in list form, length is the num_block
        # logits[-1] has the shape of B T C

        output_logits = logits[-1]#[:,-1,:] # B T C


        # --- Loss --- 
        
        # how negative sampling proceeds
        pos_embs_item = self.item_embedding(torch.LongTensor(pos_seqs).to(self.dev)) # B T C
        neg_embs_item = self.item_embedding(torch.LongTensor(neg_seqs).to(self.dev)) # B T C

        # for user embedding use above variable
        pos_embs = torch.cat([pos_embs_item,u_latent], dim =2).view(seq_.size(0),-1,self.hidden_units) 
        neg_embs = torch.cat([neg_embs_item,u_latent], dim =2).view(seq_.size(0),-1,self.hidden_units)

        # calculate loss - from STRec Backbone.py - check when loss does not change
        pos_logits = (output_logits*pos_embs).sum(dim=-1)
        neg_logits = (output_logits*neg_embs).sum(dim=-1)

        return pos_logits, neg_logits





if __name__ == '__main__':
    pass