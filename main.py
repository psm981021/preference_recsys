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

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset, SASRecDataset

from trainers import UPTRecTrainer
from models import UPTRec, OfflineItemSimilarity, OnlineItemSimilarity
from utils import *


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="data/", type=str)
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
        "--attention_type",
        default ="Base",
        type=str,
        help="Ways of performing Attention Mechanism\
            Supports 'Base' for Self-Attention and 'Cluster' for Clustered Atteniton"
    )
    parser.add_argument(
        "--num_intent_clusters",
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
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--save_pt",type=str, default="False")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=3500, help="number of epochs")
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.3, help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--device",type=str, default="cuda:1")
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}-{args.num_intent_clusters}-{args.batch_size}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    show_args_info(args)

    with open(args.log_file, "w") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # training data for node classification
    if args.contrast_type == "None":
        cluster_dataset = SASRecDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=True)

        train_dataset = SASRecDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)

        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

        test_dataset = SASRecDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=True)

    else:
        cluster_dataset = RecWithContrastiveLearningDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

        train_dataset = RecWithContrastiveLearningDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = UPTRec(args=args)
    
    trainer = UPTRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)
    
    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        args.pca = "True"
        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        start_time = time.time()

        print(f"Train ICLRec")
        early_stopping = EarlyStopping(args.log_file,args.checkpoint_path, args.patience, verbose=True)
        args.start_epochs = 0
        if os.path.exists(args.checkpoint_path):
            #print("continue learning");import IPython; IPython.embed(colors='Linux');exit(1)
            trainer.load(args.checkpoint_path)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                save_epoch = epoch
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

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
        f.write(f"To run Epoch:{save_epoch} , It took {hours} hours, {minutes} minutes, {seconds} seconds\n")


main()


# test
#python main.py --model_idx="test_1" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=16 --gpu_id=0 --attention_type="Cluster" --epochs=10


# ------- baseline model can be used as SASRec ----------
# python main.py --model_idx="SASRec" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=1 --gpu_id=1

# eval 
# python main.py --model_name="ICLRec" --model_idx="Baseline" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=1 --gpu_id=1 --do_eval

#continue training 
# python main.py --model_idx="SASRec" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=1 --gpu_id=1 --save_pt="True"


# ------- Cluster Attention for UPTRec ---------
# run version - clustering using item embedding
# python main.py --model_name="UPTRec" --model_idx="UPTRec_Clustered_Attention" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=16 --gpu_id=0 --attention_type="Cluster"

# run version - Clustering using encoder
#python main.py --model_idx="UPTRec_Clustered_Attention_encoder" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=16 --gpu_id=0 --attention_type="Cluster"

# eval
# python main.py --model_idx="UPTRec_Clustered_Attention_item_embedding" --contrast_type="None" --seq_representation_type="concatenate" --num_intent_clusters=16 --gpu_id=0 --attention_type="Cluster" --do_eval
# ---------- Amazon Beauty --------------

# Basline 
# scripts/Beauty/Baseline.sh

# Clustered Attention Version - epoch 3500
# scripts/Beauty/Cluster_Attention.sh    

# ---------- Amazon Toys_and_Games -------------------

# Clustered Attention Version - epoch 3500
# scripts/Toys_and_Games/Cluster_Attention.sh    



# ----------- Amazon Sports_and_Outdoors ---------------

# Clustered Attention Version - epoch 3500
# scripts/Sports_and_Outdoors/Cluster_Attention.sh  