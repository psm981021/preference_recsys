import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from kmeans_torch import kmeans, kmeans_predict
from modules import Encoder

import faiss


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            import IPython; IPython.embed(colors='Linux');exit(1)
            self.clus.train(x, self.index)

        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]

class UPTRec(torch.nn.Module):
    def __init__(self, args):
        super(UPTRec, self).__init__()
        self.args =args 

        self.itemnum = args.item_size
        self.usernum = args.user_size
        self.dev = args.device
        self.hidden_units = args.item_hidden_units + args.user_hidden_units
        
        self.item_embedding = nn.Embedding(self.itemnum,args.item_hidden_units)
        self.user_embedding = nn.Embedding(self.usernum,args.user_hidden_units) 
        self.position_embedding = nn.Embedding(args.maxlen, args.item_hidden_units)

        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.layernorm = nn.LayerNorm(args.item_hidden_units, eps=1e-8)

        self.encoder = Encoder(args)

        self.criterion = nn.BCELoss(reduction="none")
        self.loss_ce = nn.CrossEntropyLoss()

    def UPTembedding(self,input_ids, flag=str):

        if flag == 'predict':
            seq_emb = self.item_embedding(torch.LongTensor(input_ids).to(self.dev))
            u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).repeat(seq_emb.size(0),1)
            #u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).unsqueeze(1).repeat(1,seq_emb.size(0),1)
            
            seq_emb = torch.cat([seq_emb,u_latent], dim =1)#.view(seq_emb.size(0),-1,self.hidden_units ) # item_idx x C
            seq_emb_wop = seq_emb
        
        else: 
            
            sequence_length = input_ids.size(1)
            position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            item_embedding = self.item_embedding(input_ids)
            #user_embedding = self.user_embedding(user_ids).unsqueeze(1).repeat(1,sequence_length,1) 
            position_embedding = self.position_embedding(position_ids)
            
            sequence_embedding = item_embedding + position_embedding
            #sequence_embedding = torch.cat([item_embedding, user_embedding], dim=2) + position_embedding
            sequence_embedding = self.layernorm(sequence_embedding)
            sequence_embedding = self.emb_dropout(sequence_embedding)

            

            return sequence_embedding

            import IPython; IPython.embed(colors='Linux');exit(1)
            seq_emb = self.item_embedding(torch.LongTensor(input_ids).to(self.dev))

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


    def forward(self,input_ids,args):
        attention_mask = (input_ids > 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_embedding = self.UPTembedding(input_ids)
        encoder_layer = self.encoder(sequence_embedding, extended_attention_mask,args)
        sequence_output = encoder_layer[-1]

        return sequence_output 

    def forward_before(self,user_ids, seq, pos_seqs, neg_seqs,args):

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

        # Brodacast in last dim
        seq_emb *= ~timeline_mask.unsqueeze(-1) 

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
        logits = self.encoder(seq_emb ,attention_mask, timeline_mask,args) # logits contains in list form, length is the num_block
        # logits[-1] has the shape of B x T x C

        output_logits = logits[-1]#[:,-1,:] # B x T x C

        return output_logits




if __name__ == '__main__':
    #python main.py --dataset=Beauty --train_dir=test
    pass