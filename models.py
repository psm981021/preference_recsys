import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class SASRec(torch.nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init()
        self.itemnum = itemnum
        self.usernum = usernum
        self.dev = args.device

        self.item_embedding = nn.Embedding(self.itemnum+1,args.hidden_units) #padding_idx = 0 사용?
        self.user_embedding = nn.Embedding(usernum+1,args.hidden_units) #user를 임베딩 하는 방법?
        self.position_embedding = nn.Embedding(args.maxlen, args.hidden_units*2 ) #길이 확인

        self.u = Variable(torch.LongTensor([])) # for user embedding
    

    def forward(self, input_seq, u):
        #item embedding 
        seq = self.item_embedding(torch.LongTensor(input_seq).to(self.dev))

        #positional encoding
        #positions = np.tile(np.array(range(input_seq.shape[1])), [input_seq.shape[0], 1])
        positions = torch.arange(input_seq.size(1)).unsqueeze(0).expand(input_seq.size(0),-1).to(self.dev)
        import IPython; IPython.embed(colors='Linux'); exit(1)
        
        positions = self.position_embedding()





if __name__ == '__main__':
    # dataset
    pass