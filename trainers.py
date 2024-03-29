import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from models import KMeans
from utils import recall_at_k, ndcg_k, get_metric

class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.num_intent_clusters = [self.args.cluster_num]
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            cluster = KMeans(
                num_cluster=num_intent_cluster,
                seed=self.args.seed,
                hidden_size=self.args.item_hidden_units + args.user_hidden_units, #check for validity 
                gpu_id=int(self.args.device.split(':')[-1]),
                device=self.device,
            )
            self.clusters.append(cluster)

        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        #self.cf_criterion = NCELoss(self.args.temperature, self.device)
        #self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@15": "{:.4f}".format(recall[2]),
            "NDCG@15": "{:.4f}".format(ndcg[2]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)

        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1],recall[2], ndcg[2], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        #check for file_name
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        #check for file_name
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids,user_id):
        sequence_length = seq_out.size(1)

        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embedding(pos_ids)
        neg_emb = self.model.item_embedding(neg_ids)

        # pos_user_emb = self.model.user_embedding(user_id).unsqueeze(1).repeat(1,sequence_length,1)
        # neg_user_emb = self.model.user_embedding(user_id).unsqueeze(1).repeat(1,sequence_length,1)

        # pos_emb = torch.cat([pos_emb,pos_user_emb], dim=2)
        # neg_emb = torch.cat([neg_emb,neg_user_emb], dim=2)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        
        
        #hidden_units = self.args.item_hidden_units+self.args.user_hidden_units
        seq_emb = seq_out.view(-1, self.args.item_hidden_units)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.maxlen).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embedding(test_neg_sample)

        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        #hidden_size = int(self.model.item_embedding.weight.size(1))


        test_item_emb = self.model.item_embedding.weight 
        #test_user_emb = self.model.user_embedding.weight

        # [batch hidden_size ]
        rating_pred_item = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        #rating_pred_user = torch.matmul(seq_out[:,hidden_size:], test_user_emb.transpose(0, 1))

        rating_pred = rating_pred_item 
        
        return rating_pred


class UPTRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(UPTRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )


    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        if train:
            # ------ intentions clustering ----- #
            if epoch >= self.args.warm_up_epoches:
                print("Preparing Clustering:")
                self.model.eval()
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
    
                for i, (rec_batch) in rec_cf_data_iter:
                #for i, (rec_batch, _) in rec_cf_data_iter:
                    
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    
                    user_id, input_ids, target_pos, target_neg, answer = rec_batch

                    
                    sequence_output = self.model(input_ids,self.args)
                    
                    sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                    sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                    sequence_output = sequence_output.detach().cpu().numpy()
                    kmeans_training_data.append(sequence_output)
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                
                # train multiple clusters
                print("Training Clusters:")
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster
                # clean memory
                del kmeans_training_data
                import gc

                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            
            for i, (rec_batch) in rec_cf_data_iter:
            #for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                """
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """
                
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                
                #_, input_ids, target_pos, target_neg, _ = rec_batch
                user_id, input_ids, target_pos, target_neg, answer = rec_batch
                # ---------- recommendation task ---------------#
                sequence_output = self.model(input_ids,self.args)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg,user_id)
                
                self.optim.zero_grad()
                rec_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids,self.args)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
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
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
