# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import os
import numpy as np
import random
import torch
import argparse
import time
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import loralib as lora

from Datasets import RecWithContrastiveLearningDataset, SASRecDataset, ItemembeddingDataset
from trainers import UPTRecTrainer
from trainers_2 import UPTRecTrainer_pre
from models import UPTRec
from utils import *

import wandb

def show_args_info(args, log_file=None):
    print("Arguments:")
    for arg in vars(args):
        value = getattr(args, arg)  # Get the value of the argument

        # Convert torch.device objects to string
        if isinstance(value, torch.device):
            value = str(value)
        if isinstance(value, (list, tuple, dict)):
            value = str(value)
        # Handle user_list argument
        if arg == "user_list":
            value_str = ', '.join(map(str, value)) if value else '[]'
            print(f"{arg:<30} : {value_str:>35}")
        else:
            # Use the value variable instead of calling getattr again
            try:
                print(f"{arg:<30} : {value:>35}")
            except:
                import IPython; IPython.embed(colors='Linux');exit(1);
def set_device(args):
    if not torch.cuda.is_available() or args.no_cuda:
        return torch.device('cpu')
    if args.use_multi_gpu:
        return torch.device(f"cuda:{args.multi_devices.split(',')[0]}")  # Default to first GPU in list
    return torch.device(f"cuda:{args.gpu_id}")


def main():

    parser = argparse.ArgumentParser()

    #parser added
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--description", action="store_true")
    parser.add_argument("--attention_map", action="store_true")
    parser.add_argument("--vanilla_attention", action="store_true",help="whether to use two blocks for key")
    parser.add_argument("--alignment_loss", action="store_true", help="Alignment Loss from SimCLR.")
    parser.add_argument("--wandb", action="store_true", help="activate wandb.")
    parser.add_argument("--valid_attention", action="store_true", help="valid,test on self-attention if activated.")
    parser.add_argument("--cluster_valid", action="store_true", help="do not perform Cluster Attention in valid,test if activated.")
    parser.add_argument("--softmax", action="store_true", help="softmax after cluster attention.")
    parser.add_argument("--cluster_joint", action="store_true", help="use cluster attention as a auxilary information.")
    parser.add_argument("--cluster_prediction", action="store_true", help="use cluster prediction loss.")
    parser.add_argument("--cluster_temperature", action="store_true", help="use density as cluster temperature")
    parser.add_argument("--mlp", action="store_true", help="adapt mlp for cluster head")    
    parser.add_argument("--ncl", action="store_true", help="non negative vector for similarity")   
    parser.add_argument("--simclr", action="store_true", help="non negative vector for similarity")   
    parser.add_argument("--bi_direction", action="store_true", help="bi-directional attention mask") 

    parser.add_argument("--pre_train", action="store_true", help="pre-training for cluster-attention &  fine-tuning for contrastive learning ")  
    parser.add_argument("--fine_tune", action="store_true", help="pre-training for cluster-attention &  fine-tuning for contrastive learning ")   


    parser.add_argument(
        "--attention_type",
        default ="Base",
        type=str,
        help="Ways of performing Attention Mechanism\
            Supports 'Base' for Self-Attention and 'Cluster' for Clustered Atteniton"
    )
    parser.add_argument(
        "--context",
        default ="item_embedding",
        type=str,
        help="Ways of considering contexutal information using input_ids \
            Supports 'item_embedding' for low-dimensional representations and 'encoder' for high-dimensional representations"
    )

    parser.add_argument("--cluster_train", default=1, type=int)
    parser.add_argument("--cluster_attention_type", default=0, type=int, help='Type of Cluster-Attention \
                        Supports 0 as Block-wise, concatenated Cluster attention C // K \
                        Supports 1 as Block-wise concatenated Cluster attention for only same Cluster-ID \
                        Supports 2 as using Mask for Cluster attention, for different Cluster-ID assign -inf')
    parser.add_argument("--visualization_epoch", default=50, type=int)
    parser.add_argument('--user_list', nargs='+', default=[], help='List to store user data')


    # system args
    parser.add_argument("--data_dir", default="data/2014/", type=str)
    parser.add_argument("--output_dir", default="output_custom/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default="test", type=str, help="model idenfier")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    
    # data augmentation args
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )
    parser.add_argument(
        "--training_data_ratio",
        default=1.0,
        type=float,
        help="percentage of training samples used for training - robustness analysis",
    )
    parser.add_argument(
        "--augment_type",
        default="random",
        type=str,
        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).",
    )
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )
    parser.add_argument(
        "--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied."
    )
    parser.add_argument(
        "--contrast_type",
        default ="IntentCL",
        #default="Hybrid",
        type=str,
        help="Ways to contrastive of. \
                        Support InstanceCL and ShortInterestCL, IntentCL, and Hybrid types, None",
    )

    parser.add_argument(
        "--num_intent_clusters",
        default="256",
        type=str,
        help="Number of cluster of intents. Activated only when using \
                        IntentCL or Hybrid types.",
    )
    parser.add_argument(
        "--num_user_intent_clusters",
        default="256",
        type=str,
        help="Number of cluster of intents. Activated only when using \
                        IntentCL or Hybrid types.",
    )
    parser.add_argument(
        "--seq_representation_type",
        default="mean",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument(
        "--seq_representation_instancecl_type",
        default="concatenate",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start IntentCL.")
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")

    # model args
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_user_weight", type=float, default=0.1, help="weight of user-level contrastive learning task")
    parser.add_argument("--align_weight", type=float, default=0.001, help="weight of contrastive learning task")
    parser.add_argument("--cluster_value", type=float, default=0.1, help="value for cluster-attention joint learning")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--position_encoding_false', action='store_true', help='deactivate position encoding', default=False)
    parser.add_argument('--multi_devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    if args.embedding:
        embedding_path = os.path.join(args.output_dir, "embedding")
        check_path(embedding_path)
    
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    try:
        args.data_file = f'{args.data_dir}/{args.data_name}/{args.data_name}_seq.txt'
        user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    except: 
        args.data_file = f'{args.data_dir}/{args.data_name}.txt'
        user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    args.device = set_device(args)

    if args.description:
        description_embeddings = pd.read_csv('/home/seongbeom/paper/preference_rec/data/2014/Beauty/Beauty_embeddings.csv')
        description_embeddings = description_embeddings.sort_values(by='asin')
        description_embeddings = torch.tensor(description_embeddings.drop(columns=['asin']).values, dtype=torch.float32)
        model = UPTRec(args=args, description_embedding = description_embeddings)
    else:
        model =UPTRec(args=args)

    import IPython; IPython.embed(colors='Linux');exit(1);
        
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        args.device_ids = list(map(int, args.multi_devices.split(',')))
        model = nn.DataParallel(model, device_ids=args.device_ids)

    # user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    item_ids = torch.arange(0, args.item_size-1, dtype=torch.long).to(torch.device("cuda" if args.cuda_condition else "cpu"))

    # save model args
    args_str = f"{args.model_idx}-{args.num_intent_clusters}-{args.batch_size}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    show_args_info(args,args.log_file)

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    if os.path.exists(args.checkpoint_path):
        if args.fine_tune:
            model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
            
            checkpoint = args_str + "_fine_tune.pt"
            args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
            lora.mark_only_lora_as_trainable(model)
            args.log_file = os.path.join(args.output_dir, args_str + "_fine_tune.txt")

            with open(args.log_file, "a") as f:
                f.write("------------------------------LoRA Fine Tuning------------------------------ \n")
        with open(args.log_file, "a") as f:
            f.write("------------------------------ Continue Training ------------------------------ \n")
        
    else:
        with open(args.log_file, "a") as f:
            f.write(str(args) + "\n")



    # training data for node classification

    cluster_dataset = RecWithContrastiveLearningDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=True)

    train_dataset = RecWithContrastiveLearningDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True ,num_workers=4, pin_memory=True)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True, num_workers=4, pin_memory=True)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=True, num_workers=4, pin_memory=True)

    item_dataset = ItemembeddingDataset(item_ids)
    item_sampler = SequentialSampler(item_dataset)
    item_dataloader = DataLoader(item_dataset, sampler=item_sampler, batch_size=args.batch_size)

    if args.pre_train or args.fine_tune:
        trainer = UPTRecTrainer_pre(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader,item_dataloader, args)
    else:
        trainer = UPTRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader,item_dataloader, args)
    
    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(args.epochs, full_sort=True)

    else:
        if args.wandb == True:
            wandb.init(project="preference_rec",
                    name=f"{args.data_name}_{args.model_idx}_{args.batch_size}_{args.num_intent_clusters}_{args.epochs}",
                    config=args)
            args = wandb.config

        start_time = time.time()
        print(f"Train UPTRec")
        early_stopping = EarlyStopping(args,args.log_file,args.checkpoint_path, args.patience, verbose=True)
        if os.path.exists(args.checkpoint_path):
            print("Load pth")
            trainer.load(args.checkpoint_path)

        # saves reassignments of user cluster ID

        for epoch in range(args.epochs):
            
            trainer.train(epoch)

            # evaluate on NDCG@20
            if args.pre_train:
                scores = trainer.valid(epoch, full_sort=True)
                print(f"[Eval] MLM loss: {scores:.6f}")                
                early_stopping([scores.detach().cpu().numpy()], trainer.model)
            else:
                scores, _ = trainer.valid(epoch, full_sort=True)
                early_stopping(np.array(scores[-1:]), trainer.model)
            
            if early_stopping.early_stop:
                save_epoch = epoch
                print("Early stopping")
                break

            if args.wandb == True:
                wandb.log({
                    "HIT@5": scores[0],
                    "NDCG@5": scores[1],
                    "HIT@10": scores[2],
                    "NDCG@10": scores[3],
                    "HIT@15": scores[4],
                    "NDCG@15": scores[5],
                    "HIT@20": scores[6],
                    "NDCG@20": scores[7]}, step=epoch)
        trainer.args.train_matrix = test_rating_matrix

        print("---------------Change to test_rating_matrix!-------------------")

        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path),strict=False)
        scores, result_info = trainer.test(args.epochs, full_sort=True)

        end_time = time.time()
        execution_time = end_time - start_time

        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)

        print(args_str)
        print(result_info)
        with open(args.log_file, "a") as f:
            f.write(args_str + "\n")
            f.write(result_info + "\n")
            try:
                f.write(f"To run Epoch:{save_epoch} , It took {hours} hours, {minutes} minutes, {seconds} seconds\n")
            except:
                f.write(f"To run Epoch:{args.epochs} , It took {hours} hours, {minutes} minutes, {seconds} seconds\n")
        if args.wandb == True:
            wandb.finish()

        # convert cluster reassignment 
        if len(args.user_list) > 0:
            user_array = np.concatenate([item.numpy().astype(int) for item in args.user_list], axis=1)
            csv_file_name = f"{args.output_dir}/{args.model_idx}_cluster_reassignment.csv"
            np.savetxt(csv_file_name, user_array, delimiter=",", fmt='%d')

main()

