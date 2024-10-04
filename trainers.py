# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.clustering import NormalizedMutualInfoScore

from models import KMeans
from modules import NCELoss, NTXent, SupConLoss, PCLoss, AlignmentLossWithSinkhorn, SimCLR
from utils import *

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import wandb
import loralib as lora

class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader,item_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        # self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.num_user_intent_clusters = [int(i) for i in self.args.num_user_intent_clusters.split(",")]

        self.item_clusters =[]
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=1,
                    temperature= args.temperature,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.item_clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    temperature= args.temperature,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.item_clusters.append(cluster)


        self.clusters = []
        for num_intent_cluster in self.num_user_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    temperature= args.temperature,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * args.max_seq_length,
                    temperature= args.temperature,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        
        if self.args.seq_representation_type == 'mean':
            self.projection_user = nn.Sequential(
                nn.Linear(self.args.hidden_size , self.args.batch_size, bias=False),
                nn.BatchNorm1d(self.args.batch_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.batch_size, self.args.hidden_size, bias=True),
            )
            self.projection_item = nn.Sequential(
                nn.Linear(1, self.args.batch_size, bias=False),
                nn.BatchNorm1d(self.args.batch_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.batch_size, 1, bias=True),
            )
            
        else:
            self.projection_user = nn.Sequential(
                nn.Linear(self.args.hidden_size*self.args.max_seq_length , self.args.batch_size, bias=False),
                nn.BatchNorm1d(self.args.batch_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.batch_size, self.args.hidden_size*self.args.max_seq_length, bias=True),
            )

            self.projection_item = nn.Sequential(
                nn.Linear(self.args.hidden_size, self.args.batch_size, bias=False),
                nn.BatchNorm1d(self.args.batch_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.batch_size, self.args.hidden_size, bias=True),
            )
        if self.cuda_condition:
            self.model.cuda()
            self.projection_user.cuda()
            self.projection_item.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.item_dataloader = item_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args,self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args,self.args.temperature, self.device)
        self.simclr_criterion = SimCLR(self.args,self.args.temperature, self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.nmi = NormalizedMutualInfoScore().to(self.device)
        self.align_criterion = AlignmentLossWithSinkhorn().to(self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader, self.item_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader,self.cluster_dataloader,self.item_dataloader, full_sort=full_sort, train=False, test=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader,self.cluster_dataloader,self.item_dataloader, full_sort=full_sort, train=False, test=True)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        HIT_20, NDCG_20, MRR = get_metric(pred_list, 20)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.6f}".format(HIT_1),
            "NDCG@1": "{:.6f}".format(NDCG_1),
            "HIT@5": "{:.6f}".format(HIT_5),
            "NDCG@5": "{:.6f}".format(NDCG_5),
            "HIT@10": "{:.6f}".format(HIT_10),
            "NDCG@10": "{:.6f}".format(NDCG_10),
            "MRR": "{:.6f}".format(MRR),
            "HIT@20": "{:.6f}".format(HIT_10),
            "NDCG@20": "{:.6f}".format(NDCG_10),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
        
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20, 40]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.6f}".format(recall[0]),
            "NDCG@5": "{:.6f}".format(ndcg[0]),
            "HIT@10": "{:.6f}".format(recall[1]),
            "NDCG@10": "{:.6f}".format(ndcg[1]),
            "HIT@15": "{:.6f}".format(recall[2]),
            "NDCG@15": "{:.6f}".format(ndcg[2]),
            "HIT@20": "{:.6f}".format(recall[3]),
            "NDCG@20": "{:.6f}".format(ndcg[3]),
            "HIT@40": "{:.6f}".format(recall[4]),
            "NDCG@40": "{:.6f}".format(ndcg[4]),
        }

        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name),strict=False)

    def item_embedding_plot(self, epoch, item_dataloader):
        """
        plot learned item embedding using t-SNE for visualization
        """
        kmeans_training_data = []
        
        item_iter = tqdm(enumerate(item_dataloader), total=len(item_dataloader))
        for i, id in item_iter:
            id_ = torch.stack([t.to(self.device) for t in id], dim=0)
            
            sequence_output = self.model.item_embeddings(id_)

            sequence_output = sequence_output.view(sequence_output.shape[0], -1).detach().cpu().numpy()# B hidden
            
            kmeans_training_data.append(sequence_output)

        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) #[* hidden]
    
        tsne = TSNE(n_components=2, perplexity=30.0)
        kmeans_training_data_2d = tsne.fit_transform(kmeans_training_data)
        plt.figure(figsize=(10, 10))
        plt.scatter(kmeans_training_data_2d[:, 0], kmeans_training_data_2d[:, 1], s=5, alpha=0.7)
        plt.title('t-SNE Visualization of KMeans Training Data')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')

        x_min, x_max = np.min(kmeans_training_data_2d[:, 0]), np.max(kmeans_training_data_2d[:, 0])
        plt.xlim(x_min, x_max)
        output_file = f'{self.args.output_dir}/embedding/Item Embedding Visualization , Epoch:{epoch}.png'
        plt.savefig(output_file)
        plt.close()
        



    def embedding_plot(self,epoch,i, sequence_context, intent_ids ):
        """
        Plots the embeddings using t-SNE for visualization
        """
        sequence_context = torch.tensor(sequence_context)

        tsne = TSNE(n_components=2, perplexity=30.0)
        embedding_2d = tsne.fit_transform(sequence_context)
        intent_ids = intent_ids.int() if torch.is_tensor(intent_ids) else intent_ids.astype(int)

        plt.figure(figsize=(10, 10))
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=intent_ids.cpu(), cmap='viridis', alpha=0.5)
        plt.colorbar(label='Intent ID')
        plt.title(f'Embedding Visualization (Epoch: {epoch}, Batch: {i})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        x_min, x_max = np.min(embedding_2d[:, 0]), np.max(embedding_2d[:, 0])
        plt.xlim(x_min, x_max)
        output_file = f'{self.args.output_dir}/embedding/Embedding Visualization Batch:{i}, Epoch:{epoch}.png'
        plt.savefig(output_file)
        plt.close()


    def attention_map_plot(self, epoch, i, attention_map):
        """
        Plot attention map for the i-th sample in the batch
        """
        data = attention_map[i].detach().cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='Greens', vmin=0, vmax=0.5)

        plt.xlabel('Item index')
        plt.ylabel('Item index')
        plt.title(f'Attention Map (Epoch: {epoch}, {i}-th sample)')
        plt.colorbar()
        plt.gca().invert_yaxis()

        output_file = f'{self.args.output_dir}/Attention map {i}-th sample, Epoch:{epoch}.png'
        plt.savefig(output_file)
        plt.close()


        

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
       
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch seq_length hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class UPTRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader,item_dataloader, args):
        super(UPTRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader,item_dataloader, args
        )

    def _instance_wsie_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
    
        cl_sequence_output,_ = self.model(cl_batch,self.args)

        if self.args.seq_representation_type == "mean":
            
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)

        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

        cl_output_slice_0 = cl_output_slice[0]
        cl_output_slice_1 = cl_output_slice[1]

        if self.args.mlp:
            cl_output_slice_0 = self.projection_user(cl_output_slice[0])
            cl_output_slice_1 = self.projection_user(cl_output_slice[1])

        
        if self.args.de_noise:
            if self.args.simclr:
                cl_loss = self.simclr_criterion('user', cl_output_slice_0, cl_output_slice_1, intent_ids = intent_ids)
            else:
                cl_loss = self.cf_criterion('user',cl_output_slice_0, cl_output_slice_1, intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion('user',cl_output_slice_0, cl_output_slice_1, intent_ids=None)
        return cl_loss


    def pcl_item_pair_contrastive_learning(self, inputs, intents, intent_ids,temperature=None):
        
        
        cl_intents = intents[0].view(self.args.batch_size,self.args.max_seq_length,-1) # B x C x E
        cl_intent_ids = intent_ids[0].view(self.args.batch_size, -1) # B x C

        if temperature is not None:
            prototype_density = temperature[0].view(self.args.batch_size, -1)
        elif temperature is None:
            prototype_density = None

        inputs[0] =inputs[0].to(self.device)
        inputs[1] =inputs[1].to(self.device)

        if self.args.description:
            cl_sequence_output_view_1,_ = self.model(inputs[0],self.args,description='description')
            cl_sequence_output_view_2,_ = self.model(inputs[1],self.args,description='description')
        else:
            cl_sequence_output_view_1,_ = self.model(inputs[0],self.args,intent_ids)
            cl_sequence_output_view_2,_ = self.model(inputs[1],self.args,intent_ids)

        if self.args.seq_representation_type == 'mean':
            
            cl_sequence_output_view_1 = torch.mean(cl_sequence_output_view_1, dim=2,keepdim=False).view(cl_sequence_output_view_1.shape[0],self.args.max_seq_length,-1)
            cl_sequence_output_view_2 = torch.mean(cl_sequence_output_view_2, dim=2,keepdim=False ).view(cl_sequence_output_view_2.shape[0],self.args.max_seq_length,-1)
        
        if self.args.mlp:
            cl_sequence_output_view_1 = self.projection_item(cl_sequence_output_view_1.view(cl_sequence_output_view_1.shape[0]*self.args.max_seq_length,-1)).view(self.args.batch_size, self.args.max_seq_length,-1)
            cl_sequence_output_view_2 = self.projection_item(cl_sequence_output_view_2.view(cl_sequence_output_view_1.shape[0]*self.args.max_seq_length,-1)).view(self.args.batch_size, self.args.max_seq_length,-1)
        
        #PCLoss
        if self.args.de_noise:
            cl_loss = self.pcl_criterion('item',cl_sequence_output_view_1, cl_sequence_output_view_2, intents=cl_intents, intent_ids=cl_intent_ids,temperature=prototype_density)
        else:
            cl_loss = self.pcl_criterion(cl_sequence_output_view_1, cl_sequence_output_view_2, intents=cl_intents, intent_ids=None)
        return cl_loss


    def pcl_user_pair_contrastive_learning(self, inputs, intents, intent_ids, temperature=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        if temperature is not None:
            prototype_density = temperature[0].view(self.args.batch_size, -1)

        cl_sequence_output = self.model(cl_batch,self.args)
       
    
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)

        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)

        cl_output_slice_0 = cl_output_slice[0]
        cl_output_slice_1 = cl_output_slice[1]

        if self.args.mlp:
            cl_output_slice_0 = self.projection_user(cl_output_slice[0])
            cl_output_slice_1 = self.projection_user(cl_output_slice[1])      
            

        #PCLoss
        if self.args.de_noise:
            if temperature is not None:
                cl_loss = self.pcl_criterion('user',cl_output_slice_0, cl_output_slice_1, intents=intents, intent_ids=intent_ids, temperature = prototype_density)
            else:
                cl_loss = self.pcl_criterion('user',cl_output_slice_0, cl_output_slice_1, intents=intents, intent_ids=intent_ids)
            
        else:
            cl_loss = self.pcl_criterion('user', cl_output_slice_0, cl_output_slice_1, intents=intents, intent_ids=None)
        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, item_dataloader=None, full_sort=True, train=True, test=False):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        if train:
            # ------ intentions clustering ----- #
            if self.args.contrast_type in ["IntentCL", "Hybrid","Item-User","Item-Level","Item-description","User","None"] and epoch >= self.args.warm_up_epoches:
                print("[Train] Preparing User,Item Clustering:")
                self.model.eval()
                kmeans_training_data = []
                kmeans_training_data_item = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                
                for i, (rec_batch, _, _) in rec_cf_data_iter:
                   
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, input_ids, target_pos, target_neg, _ = rec_batch
                    
                    if self.args.context == "encoder":
                        sequence_output,_ = self.model(input_ids,self.args)
                        
                    if self.args.context == "item_embedding":
                        sequence_output,_ = self.model.item_embeddings(input_ids)

                    sequence_context_item = sequence_output.view(self.args.batch_size*self.args.max_seq_length,-1)
                    sequence_context_user = sequence_output.view(self.args.batch_size,-1)
          
                    # average sum
                    if self.args.seq_representation_type == "mean":
                
                        sequence_context_user = torch.mean(sequence_output, dim=1, keepdim=False).view(sequence_output.shape[0],-1)
                        sequence_context_item = torch.mean(sequence_output, dim=2, keepdim=False).view(sequence_output.shape[0]*self.args.max_seq_length,-1)

                        sequence_context_user = sequence_context_user.detach().cpu().numpy()
                        sequence_context_item = sequence_context_item.detach().cpu().numpy()
                        

                    # sequence_context_item = self.projection_item(sequence_context_item).detach().cpu().numpy()
                    # sequence_context_user = self.projection_user(sequence_context_user).detach().cpu().numpy()
                    
                    kmeans_training_data.append(sequence_context_user)
                    kmeans_training_data_item.append(sequence_context_item) 

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) #[* hidden]
                kmeans_training_data_item = np.concatenate(kmeans_training_data_item, axis=0) #[* hidden]
                
                # train multiple clusters
                print("[Train] Training User Clusters:")
                
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster
                
                # clean memory
                del kmeans_training_data

                print("[Train] Training Item Clusters:")
                for i, cluster in tqdm(enumerate(self.item_clusters), total=len(self.item_clusters)):
                    cluster.train(kmeans_training_data_item)
                    self.item_clusters[i] = cluster

                del kmeans_training_data_item
                import gc

                gc.collect()
                
            #train
            # ------ model training -----#
            print("Model Training")
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            nmi_assignment = []

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            print("Performing Rec model Training (UPTRec) ")
                    

            if self.args.embedding == True and epoch % self.args.visualization_epoch == 0 and epoch >0:
                self.item_embedding_plot(epoch, item_dataloader)
                # self.embedding_plot(epoch, i, sequence_context, intent_ids[0])

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                """
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                if self.args.attention_type in ["Cluster"] and epoch >= self.args.warm_up_epoches:
                    if self.args.context == "encoder":
                        sequence_context,_ = self.model(input_ids,self.args)
                    if self.args.context == "item_embedding":
                        sequence_context = self.model.item_embeddings(input_ids)
                    
                    # sequence_context = sequence_output.view(self.args.batch_size*self.args.max_seq_length,-1)

                    if self.args.seq_representation_type == "mean":
                        sequence_context = torch.mean(sequence_context, dim=2, keepdim=False).view(sequence_context.shape[0]*self.args.max_seq_length,-1)
                        sequence_context = sequence_context.detach().cpu().numpy()                

                    for cluster in self.item_clusters:
                        seq2intents = []
                        intent_ids = []
                        densitys = []
                        intent_id, seq2intent,density = cluster.query(sequence_context)

                        seq2intents.append(seq2intent)
                        intent_ids.append(intent_id)
                        densitys.append(density)
                    nmi_assignment.append(intent_ids[0])
                    

                # ---------- recommendation task ---------------#
                
                if self.args.attention_type == "Cluster" and epoch >= self.args.warm_up_epoches:
                    
                    recommendation_output,_ = self.model(input_ids,self.args,intent_ids)
                    
                else:
                    recommendation_output,_ = self.model(input_ids,self.args)
                    
                rec_loss = self.cross_entropy(recommendation_output, target_pos, target_neg)                

                # attention map visualization
                # if self.args.attention_map == True and epoch % self.args.visualization_epoch == 0 and i in [0,10,20]:
                #     attention_map = attention_map.to(self.device)
                
                # ---------- contrastive learning task -------------#
                cl_losses = []
                    
                for cl_batch in cl_batches:
                    
                    if self.args.contrast_type == "InstanceCL":
                        
                        cl_loss = self._instance_wsie_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches
                            )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":
                        # ------ performing clustering for getting users' intentions ----#
             
                        if self.args.seq_representation_type == "mean":
                            recommendation_output_ = torch.mean(recommendation_output, dim=1, keepdim=False)
                        recommendation_output_ = recommendation_output.view(recommendation_output.shape[0], -1)
                        recommendation_output_ = recommendation_output_.detach().cpu().numpy()
                        
                        # query on multiple clusters
                        for cluster in self.clusters:
                            seq2intents = []
                            intent_ids = []
                            intent_id, seq2intent,density = cluster.query(recommendation_output_)
                            seq2intents.append(seq2intent)
                            intent_ids.append(intent_id)
                        cl_loss = self.pcl_user_pair_contrastive_learning(
                            cl_batch, intents=seq2intents, intent_ids=intent_ids
                        )
                        cl_losses.append(self.args.intent_cf_weight * cl_loss)

                        
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_wsie_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                        else:
                            cl_loss1 = self._instance_wsie_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                            if self.args.seq_representation_type == "mean":
                                recommendation_output = torch.mean(recommendation_output, dim=1, keepdim=False)
                            
                            recommendation_output = recommendation_output.view(recommendation_output.shape[0], -1)
                            recommendation_output = recommendation_output.detach().cpu().numpy()

                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                densitys = []
                            
                                intent_id, seq2intent,density = cluster.query(recommendation_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                                densitys.append(density)

                            cl_loss3 = self.pcl_user_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids,temperature=densitys
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)

                    elif self.args.contrast_type == "Item-Level":

                        if epoch >= self.args.warm_up_epoches:
                            
                            cl_loss1 = self._instance_wsie_one_pair_contrastive_learning(
                                    cl_batch, intent_ids=seq_class_label_batches
                                )
                            cl_losses.append(self.args.cf_weight * cl_loss1)

                            if self.args.seq_representation_type == "mean":
                                
                                sequence_context = torch.mean(recommendation_output, dim=2, keepdim=False).view(recommendation_output.shape[0],-1)
                                sequence_output = sequence_context.view(self.args.batch_size*self.args.max_seq_length, -1).detach().cpu().numpy()                
                            
                            # sequence_output = self.projection_item(recommendation_output.view(recommendation_output.shape[0]*self.args.max_seq_length, -1)).detach().cpu().numpy()
                            # sequence_output = recommendation_output.view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                            
                            for cluster in self.item_clusters:
                                seq2intents = []
                                intent_ids = []
                                densitys = []
                                intent_id, seq2intent,density = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                                densitys.append(density)

                            if self.args.cluster_temperature:
                                cl_loss3 = self.pcl_item_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids,temperature=densitys
                                )
                            else: 
                                cl_loss3 = self.pcl_item_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids
                                )
                            
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)
                            
                            if self.args.cluster_prediction:
                                input_1 = cl_batch[0].to(self.device)
                                input_2 = cl_batch[1].to(self.device)

                                cl_output_1,_ = self.model(input_1,self.args, intent_ids)
                                cl_output_2,_ = self.model(input_2,self.args, intent_ids)
                                # cl_output_1 = self.model(input_1,self.args, intent_ids).view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                                # cl_output_2 = self.model(input_2,self.args, intent_ids).view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                                if self.args.seq_representation_type == "mean":
                                    cl_output_1 = torch.mean(cl_output_1, dim=2, keepdim=False).view(recommendation_output.shape[0],-1)
                                    cl_output_1 = cl_output_1.view(self.args.batch_size*self.args.max_seq_length, -1).detach().cpu().numpy()  

                                    cl_output_2 = torch.mean(cl_output_2, dim=2, keepdim=False).view(recommendation_output.shape[0],-1)
                                    cl_output_2 = cl_output_2.view(self.args.batch_size*self.args.max_seq_length, -1).detach().cpu().numpy()
                                else:
                                    cl_output_1 = cl_output_1.view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                                    cl_output_2 = cl_output_2.view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy() 

                                ce_loss = 0
                                for cluster in self.item_clusters:
                                    
                                    intent_ids_1 = []
                                    intent_ids_2 = []
                                    intent_id, seq2intent,density = cluster.query(cl_output_1)
                                    intent_ids_1.append(intent_id)
                                    intent_id, seq2intent,density = cluster.query(cl_output_2)
                                    intent_ids_2.append(intent_id)
                                
                                
                                cross_entropy_loss_fn = nn.CrossEntropyLoss()
                                intent_1_logits = intent_ids_1[0].float().unsqueeze(0) 
                                intent_2_logits = intent_ids_2[0].float().unsqueeze(0) 
                                intent_labels = intent_ids[0].float().unsqueeze(0)
                                
                                ce_loss_1 = cross_entropy_loss_fn(intent_1_logits, intent_labels) 
                                ce_loss_1 /= int(self.args.max_seq_length) * int(self.args.batch_size)
                                ce_loss_2 = cross_entropy_loss_fn(intent_2_logits, intent_labels)
                                ce_loss_2 /= int(self.args.max_seq_length)  * int(self.args.batch_size)

                                ce_loss = (ce_loss_1 + ce_loss_2 ) /2

                                # cl_losses.append(ce_loss)
                                cl_losses.append(ce_loss * self.args.intent_cf_weight * 0.1)
                            
                            
                    elif self.args.contrast_type == "User":
                        cl_loss1 = self._instance_wsie_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                        cl_losses.append(self.args.cf_weight * cl_loss1)

                        # sequence_output = recommendation_output.view(recommendation_output.shape[0], -1).detach().cpu().numpy()
                        # sequence_output = self.projection_user(recommendation_output.view(recommendation_output.shape[0], -1)).detach().cpu().numpy()
                        
                        if self.args.seq_representation_type == "mean":
                            recommendation_output = torch.mean(recommendation_output, dim=1, keepdim=False)

                        recommendation_output = recommendation_output.view(recommendation_output.shape[0], -1)
                        recommendation_output = recommendation_output.detach().cpu().numpy()
                    
                        for cluster in self.clusters:
                            seq2intents = []
                            intent_ids = []
                            densitys = []
                        
                            intent_id, seq2intent,density = cluster.query(recommendation_output)
                            seq2intents.append(seq2intent)
                            intent_ids.append(intent_id)
                            densitys.append(density)
                            
                        if self.args.cluster_temperature:
                            cl_loss3 = self.pcl_user_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids,temperature = densitys
                            )
                        else:
                            cl_loss3 = self.pcl_user_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                        cl_losses.append(self.args.intent_cf_weight * cl_loss3)
                        
                        if self.args.cluster_prediction:
                            cl_1 =cl_batch[0].to(self.device)
                            cl_1_output,_ = self.model(cl_1,self.args).view(cl_1.shape[0], -1).detach().cpu().numpy()

                            cl_2 =cl_batch[1].to(self.device)
                            cl_2_output,_ = self.model(cl_2,self.args).view(cl_2.shape[0], -1).detach().cpu().numpy()

                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids_1 = []
                                intent_ids_2 = []
                                densitys = []

                                intent_id_1_, seq2intent, density = cluster.query(cl_1_output)
                                intent_id_2_, seq2intent, density = cluster.query(cl_2_output)

                                seq2intents.append(seq2intent)
                                intent_ids_1.append(intent_id_1_)
                                intent_ids_2.append(intent_id_2_)
                                densitys.append(density)

                            cross_entropy_loss_fn = nn.CrossEntropyLoss()
                            intent_1_logits = intent_ids_1[0].float().unsqueeze(0) 
                            intent_2_labels = intent_ids_2[0].float().unsqueeze(0)
                            
                            ce_loss = cross_entropy_loss_fn(intent_1_logits, intent_2_labels) * 0.00001
                            cl_losses.append(ce_loss * self.args.intent_cf_weight)

                    elif self.args.contrast_type == "Item-User":
                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(recommendation_output, dim=1, keepdim=False)

                            cl_loss1 = self._instance_wsie_one_pair_contrastive_learning(
                                    cl_batch, intent_ids=seq_class_label_batches
                                )
                            cl_losses.append(self.args.cf_weight * cl_loss1)

                            sequence_output = sequence_output.view(sequence_output.shape[0], -1).detach().cpu().numpy()
                            
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                densitys = []
                                intent_id, seq2intent, density = cluster.query(sequence_output)

                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                                densitys.append(density)
                            if self.args.cluster_temperature:
                                cl_loss3 = self.pcl_user_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids, temperature = densitys
                                )
                            else:
                                cl_loss3 = self.pcl_user_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids
                                )
                            cl_losses.append(self.args.intent_cf_user_weight * cl_loss3)

                            
                            sequence_output = recommendation_output.view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(recommendation_output, dim=2, keepdim=False).view(recommendation_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()

                            for cluster in self.item_clusters:
                                seq2intents = []
                                intent_ids = []
                                densitys = []
                                intent_id, seq2intent, density = cluster.query(sequence_output)

                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                                densitys.append(density)
                            if self.args.cluster_temperature:
                                cl_loss4 = self.pcl_item_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids, temperature = densitys
                                )
                            else:
                                cl_loss4 = self.pcl_item_pair_contrastive_learning(
                                    cl_batch, intents=seq2intents, intent_ids=intent_ids
                                )                             
                            cl_losses.append(self.args.intent_cf_weight * cl_loss4)
                            

                    elif self.args.contrast_type == "Item-description":
                        if epoch >= self.args.warm_up_epoches:

                            description_output, _ = self.model(input_ids,self.args,description='description')
                            description_view = description_output.view(self.args.batch_size,-1)

                            align_loss = (recommendation_output.view(self.args.batch_size,-1) - description_view).norm(dim=1).pow(2).mean()
                            align_loss /= int(self.args.max_seq_length)

                            cl_losses.append(self.args.align_weight * align_loss)

                            sequence_output = description_output.view(description_output.shape[0]*self.args.max_seq_length, -1).detach().cpu().numpy()
                            
                            for cluster in self.item_clusters:
                                seq2intents = []
                                intent_ids = []
                                densitys = []

                                intent_id, seq2intent,density = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                                densitys.append(density)
                            if self.args.cluster_temperature:
                                cl_loss3 = self.pcl_item_pair_contrastive_learning(
                                        cl_batch, intents=seq2intents, intent_ids=intent_ids,temperature = densitys
                                )
                            else:
                                cl_loss3 = self.pcl_item_pair_contrastive_learning(
                                        cl_batch, intents=seq2intents, intent_ids=intent_ids
                                )                              
                            cl_losses.append(self.args.cf_weight * cl_loss3)

                joint_loss = self.args.rec_weight * rec_loss
                if self.args.contrast_type in ["IntentCL", "Hybrid", "Item-Level", "Item-User", "Item-description", "User"]:
                    for cl_loss in cl_losses:
                        joint_loss += cl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if self.args.contrast_type in ["IntentCL", "Hybrid", "Item-Level","Item-User","Item-description","User"]:
                    for i, cl_loss in enumerate(cl_losses):
                        cl_sum_avg_loss += cl_loss.item()

                joint_avg_loss += joint_loss.item()

            # end of for statements in model training

            # ----------- nmi assignemnt --------------# 
            nmi_assignment_score = 0.0 
            if epoch >0:
                for before, after in zip(self.args.nmi_assignment, nmi_assignment):
                    nmi_assignment_score += self.nmi(before, after)
                self.args.nmi_assignment_score = nmi_assignment_score


            if nmi_assignment_score == 0.0 and epoch > 0:
                nmi_assignment_score = self.args.nmi_assignment_score
            
            self.args.nmi_assignment = nmi_assignment


            if self.args.contrast_type in ["None"]:

                post_fix = {
                    "epoch": epoch,
                    "rec_avg_loss": "{:.6}".format(rec_avg_loss / len(rec_cf_data_iter))
                }

                if self.args.wandb == True:
                    wandb.log({'rec_avg_loss':rec_avg_loss / len(rec_cf_data_iter)}, step=epoch)
                
            elif self.args.contrast_type in ["Hybrid","IntentCL","Item-Level","Item-User","Item-description","User"]:

                post_fix = {
                    "epoch": epoch,
                    "rec_avg_loss": "{:.6}".format(rec_avg_loss / len(rec_cf_data_iter)),
                    "joint_avg_loss": "{:.6f}".format(joint_avg_loss / len(rec_cf_data_iter)),
                    "NMI_cluster_reassignment": "{:.6f}".format(nmi_assignment_score / len(rec_cf_data_iter)),
                }

                if self.args.wandb == True:
                    wandb.log({'rec_avg_loss':rec_avg_loss / len(rec_cf_data_iter)}, step=epoch)
                    wandb.log({'joint_avg_loss': joint_avg_loss / len(rec_cf_data_iter)}, step=epoch)
                    wandb.log({'nmi_assignment_score': nmi_assignment_score / len(rec_cf_data_iter)}, step=epoch)
            else:

                post_fix = {
                    "epoch": epoch,
                    "rec_avg_loss": "{:.6}".format(rec_avg_loss / len(rec_cf_data_iter)),
                }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else: # for valid and test
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None
            if full_sort:
                if test:
                    if self.args.contrast_type in ["IntentCL", "Hybrid","Item-User","Item-Level","User","None"] and epoch >= self.args.warm_up_epoches:
                        print("[Test] Prepare Item,User Clustering")

                        kmeans_training_data = []
                        kmeans_training_data_item = []
                        rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                        for i, (rec_batch, _, _) in rec_cf_data_iter:
                            
                            rec_batch = tuple(t.to(self.device) for t in rec_batch)
                            _, input_ids, target_pos, target_neg, _ = rec_batch

                            if self.args.context == "encoder":
                                sequence_output, _ = self.model(input_ids,self.args)
                            if self.args.context == "item_embedding":
                                sequence_output = self.model.item_embeddings(input_ids)
                                    
                            sequence_context_item = sequence_output.view(self.args.batch_size*self.args.max_seq_length,-1)
                            sequence_context_user = sequence_output.view(self.args.batch_size,-1)
                            
                            if self.args.seq_representation_type == "mean":
                                sequence_context_user = torch.mean(sequence_output, dim=1, keepdim=False).view(sequence_output.shape[0],-1)
                                sequence_context_item = torch.mean(sequence_output, dim=2, keepdim=False).view(sequence_output.shape[0]*self.args.max_seq_length,-1)

                                sequence_context_user = sequence_context_user.detach().cpu().numpy()
                                sequence_context_item = sequence_context_item.detach().cpu().numpy()
                                
                        
                            kmeans_training_data.append(sequence_context_user)
                            kmeans_training_data_item.append(sequence_context_item)

                        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) #[* hidden]
                        kmeans_training_data_item = np.concatenate(kmeans_training_data_item, axis=0) #[* hidden]
                        
                        # train multiple clusters
                        print("[Test] Training User Clusters:")
                        for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                            cluster.train(kmeans_training_data)
                            self.clusters[i] = cluster
                        # clean memory
                        del kmeans_training_data

                        print("[Test] Training Item Clusters:")
                        for i, cluster in tqdm(enumerate(self.item_clusters), total=len(self.item_clusters)):
                            cluster.train(kmeans_training_data_item)
                            self.item_clusters[i] = cluster
                            
                        # clean memory
                        del kmeans_training_data_item

                        import gc
                        gc.collect()
                
                answer_list = None
                print("Model Eval ")

                # -------------perfrom valid, test on cluster-attention-------------- #
                for i, (batch,_) in rec_data_iter:
                    
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch

                    if self.args.attention_type in ["Cluster"] and epoch >= self.args.warm_up_epoches:

                        if self.args.context == "encoder":
                            sequence_context, _ = self.model(input_ids,self.args)
                        if self.args.context == "item_embedding":
                            sequence_context = self.model.item_embeddings(input_ids)

                        if self.args.seq_representation_type == "mean":
                            sequence_context = torch.mean(sequence_context, dim=2, keepdim=False).view(sequence_context.shape[0],-1)
                            sequence_context = sequence_context.view(self.args.batch_size*self.args.max_seq_length, -1).detach().cpu().numpy()   
                        
                        # sequence_context = sequence_context.view(sequence_context.shape[0], -1).detach().cpu().numpy()
                        # sequence_context = sequence_context.view(self.args.batch_size*self.args.max_seq_length,-1).detach().cpu().numpy()
                        
                        for cluster in self.item_clusters:
                            seq2intents = []
                            intent_ids = []
                            densitys = []

                            intent_id, seq2intent,density = cluster.query(sequence_context)
                            seq2intents.append(seq2intent)
                            intent_ids.append(intent_id)
                            densitys.append(density)
                        recommend_output,_ = self.model(input_ids,self.args,intent_ids)
                    
                    else:
                        recommend_output,_ = self.model(input_ids, self.args)
                        
                    recommend_output = recommend_output[:, -1, :]


                    # recommendation results
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    # ind = np.argpartition(rating_pred, -20)[:, -20:]
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    for cluster in self.clusters:
                        seq2intents = []
                        intent_ids = []
                        intent_id, seq2intent,density = cluster.query(sequence_output)
                        seq2intents.append(seq2intent)
                        intent_ids.append(intent_id)

                    recommend_output = self.model.finetune(input_ids,self.args,intent_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
