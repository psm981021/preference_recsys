import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm
import random
import faiss
import pickle
from tqdm import tqdm
import copy

class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size,temperature, gpu_id=0, device="cpu"):
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
        self.temperature = temperature
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

    # 원본 train code 5/8/24
    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)

        # get cluster centroids [num_cluster, *]
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
            
    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))

        ### code for Density ###
        # https://github.com/salesforce/PCL/blob/master/main_pcl.py#L406
        Dcluster = [[] for c in range(self.num_cluster)]
        for im,i in enumerate(seq2cluster):
            Dcluster[i].append(D[im][0])
        
        density = np.zeros(self.num_cluster)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d   
        
         #if cluster only has one point, use the max to estimate its concentration 
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 
        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = self.temperature*density/density.mean()  #scale the mean to temperature 
        density = torch.Tensor(density).to(self.device)
        
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster],density[seq2cluster]


    
class UPTRec(nn.Module):
    def __init__(self, args, description_embedding = None):
        super(UPTRec, self).__init__()

        self.itemnum=args.item_size
        self.args = args    
        self.item_embeddings = nn.Embedding(self.itemnum, args.hidden_size)
        if description_embedding is not None:
            # Create an embedding layer initialized with the description_embedding tensor
            self.description_embeddings = nn.Embedding.from_pretrained(
                description_embedding, freeze=False  
            )
        self.transform_layer =nn.Linear(384,self.args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.criterion = nn.BCELoss(reduction="none")
        self.lm_head = nn.Linear(args.hidden_size, args.item_size, bias=False)
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence,description=None):
        
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        
        if description is not None:
            item_embeddings = self.description_embeddings(sequence)
            item_embeddings = self.transform_layer(item_embeddings)
        else:
            item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids, args, cluster_id =None, description=None):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
 
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)# fp16 compatibility   
        if not self.args.bi_direction:  
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        sequence_emb = self.add_position_embedding(input_ids,description)
        
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask,args,cluster_id, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        
        prediction_scores = self.lm_head(sequence_output)
        return sequence_output, prediction_scores


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)

        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




if __name__ == '__main__':
    #python main.py --dataset=Beauty --train_dir=test
    pass