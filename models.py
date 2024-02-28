import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import plot_blobs
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.decomposition import PCA

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

    def forward(self,user_ids, seq, pos_seqs, neg_seqs):
        '''
        B : Batch
        T : seq length
        C : hidden_unit
        '''
        #--- embedding --- 

        #item embedding 
        seq = self.item_embedding(torch.LongTensor(seq).to(self.dev))

        #positional encoding
        positions = torch.arange(seq.size(1)).unsqueeze(0).expand(seq.size(0),-1)

        #position embedding
        positions = self.position_embedding(positions.to(self.dev))

        #user embedding  B x C to B x T x C
        u_latent = self.user_embedding(torch.LongTensor(user_ids).to(self.dev)).unsqueeze(1).repeat(1,seq.size(1),1) 

        # concat user embedding with item embedding
        seq = torch.cat([seq,u_latent], dim =2).view(seq.size(0),-1,self.hidden_units )
        seq += positions
        
        #dropout
        seq = self.emb_dropout(seq)
        
        
        
        #---kmeans pytorch module--- change to class if needed

        # In batch
        batch_cluster_ids =[]
        for batch in range(seq.size(0)):
            seq_cluster = seq[batch]

            seq_cluster_id, cluster_centers = kmeans(
                X=seq_cluster,
                num_clusters= 10, 
                distance = 'euclidean', 
                #iter_limit = 20,
                device = self.dev
            )
            seq_cluster_id = seq_cluster_id.to(self.dev)
            cluster_centers = cluster_centers.to(self.dev) # check whether values have been change every batch

            batch_cluster_ids.append(seq_cluster_id.view(-1,1))



        import IPython; IPython.embed(colors='Linux'); exit(1)
        
        #--- cluster mask ---


        #--- attention layer ---

        import IPython; IPython.embed(colors='Linux'); exit(1)




if __name__ == '__main__':
    # dataset
    pass