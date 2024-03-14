import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

    def UPTembedding(self,user_ids, seq, flag=str):

        if flag == 'predict':
            seq_emb = self.item_embedding(torch.LongTensor(seq).to(self.dev))
            u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).repeat(seq_emb.size(0),1)
            #u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).unsqueeze(1).repeat(1,seq_emb.size(0),1)
            
            seq_emb = torch.cat([seq_emb,u_latent], dim =1)#.view(seq_emb.size(0),-1,self.hidden_units ) # item_idx x C
            seq_emb_wop = seq_emb
        
        else: 

            # item embedding
            seq_emb = self.item_embedding(torch.LongTensor(seq).to(self.dev))

            # user embedding
            u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).unsqueeze(1).repeat(1,seq_emb.size(1),1) 

            # position encdoing
            positions = torch.arange(seq_emb.size(1)).unsqueeze(0).expand(seq_emb.size(0),-1)

            # positon embedding
            positions = self.position_embedding(positions.to(self.dev))

            # concat user embedding with item embedding
            seq_emb_wop = seq_emb
            seq_emb = torch.cat([seq_emb,u_latent], dim =2).view(seq_emb.size(0),-1,self.hidden_units )
            seq_emb += positions

            # dropout
            seq_emb = self.emb_dropout(seq_emb)

        return seq_emb, seq_emb_wop, u_latent


    def log2kmeans(self, seq, args):
        #### ---kmeans pytorch module --- ###
        batch_cluster_ids =[]

        for batch in range(seq.size(0)):
            seq_cluster = seq[batch]
            
            seq_cluster_id, cluster_centers = kmeans(
                X=seq_cluster,
                num_clusters= int(args.cluster_num), 
                distance = 'euclidean', 
                tqdm_flag=False,
                device = self.dev
                
                
            )
            seq_cluster_id = seq_cluster_id.to(self.dev)

            # check whether values have been change every batch
            cluster_centers = cluster_centers.to(self.dev) 

            batch_cluster_ids.append(seq_cluster_id.view(-1,1))

        return batch_cluster_ids
    
    def log2feats(self, user_ids, seq, args):
        '''
        B : Batch
        T : seq length
        C : hidden_unit
        '''

        seq_emb,seq_emb_wop,u_latent = self.UPTembedding(user_ids, seq)
        tl = seq_emb.shape[1] #T
        ### --- cluster --- ###
        batch_cluster_ids = self.log2kmeans(seq_emb,args)

        ### --- timeline masking --- ###
        timeline_mask = torch.BoolTensor(seq == 0).to(self.dev)
        seq_emb *= ~timeline_mask.unsqueeze(-1) # Brodacast in last dim

        ### --- attention masking using cluster ids --- ###
        
        if args.attention_mask == 'base':
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        elif args.attention_mask == 'cluster':
            #attention mask using cluster_ids
            batch_cluster_ids_tensor = torch.stack(batch_cluster_ids) # B x T x 1
            
            # Calculate a mask where each element indicates if it belongs to the same cluster as each other element
            
            #expand batch_cluster_ids for comparison
            expanded_ids = batch_cluster_ids_tensor.expand(-1, -1, seq_emb.size(1))

            # Compare cluster IDs across all positions in a sequence
            same_cluster_mask = expanded_ids == expanded_ids.transpose(1, 2)
            same_cluster_mask *= ~timeline_mask.unsqueeze(-1)
            attention_mask = ~same_cluster_mask
        else:
            print('wrong parser for attention_mask')

        ### --- attention layer --- ### 
        logits = self.encoder(seq_emb ,attention_mask, timeline_mask) # logits contains in list form, length is the num_block
        # logits[-1] has the shape of B x T x C

        output_logits = logits[-1]#[:,-1,:] # B x T x C

        return output_logits





    def forward(self,user_ids, seq, pos_seqs, neg_seqs,args):

        output_logits = self.log2feats(user_ids, seq,args)
        

        ### --- Loss --- ###
        seq_emb,seq_emb_wop,u_latent = self.UPTembedding(user_ids,seq)

        # check how negative sampling proceeds
        pos_embs_item = self.item_embedding(torch.LongTensor(pos_seqs).to(self.dev)) # B x T x C
        neg_embs_item = self.item_embedding(torch.LongTensor(neg_seqs).to(self.dev)) # B x T x C

        # for user embedding use above variable
        pos_embs = torch.cat([pos_embs_item,u_latent], dim =2).view(seq_emb.size(0),-1,self.hidden_units) 
        neg_embs = torch.cat([neg_embs_item,u_latent], dim =2).view(seq_emb.size(0),-1,self.hidden_units)

        # calculate loss - from STRec Backbone.py - check when loss does not change
        pos_logits = (output_logits*pos_embs).sum(dim=-1)
        neg_logits = (output_logits*neg_embs).sum(dim=-1)

        return pos_logits, neg_logits
    
   
    def predict(self, user_ids, seq, item_indices,args):
        output_logits = self.log2feats(user_ids,seq,args) # 1 x T x C
        
        seq_emb,seq_emb_wop,u_latent = self.UPTembedding(user_ids, item_indices, flag='predict')
        final_logits = output_logits[:,-1,:] # 1 x C

        # user embedding concat item embedding 
        logits = seq_emb.matmul(final_logits.unsqueeze(-1)).squeeze(-1)  # 1 x item_idx

        return logits





if __name__ == '__main__':
    #python main.py --dataset=Beauty --train_dir=test
    pass