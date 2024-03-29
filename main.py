import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import *
from models import UPTRec
from datasets import SequentialDataset
from trainers import UPTRecTrainer;


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def ndcg_k_type(value):
    try:
        k_values = list(map(int, value.split(',')))
        return k_values
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid NDCG@K values. Please provide a comma-separated list of integers.")
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help ="name of the dataset name")
parser.add_argument('--train_dir', required=True, help = "dir where log will be stored")
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
#parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--item_hidden_units', default= 64, type=int, help="hidden units for item embedding")
parser.add_argument('--user_hidden_units', default= 64, help ="hidden units for user embedding")
parser.add_argument('--cluster_num', default =10, help ="number of clusters") # check for intent
parser.add_argument('--threshold_user', default= 1.0, help ="threshold for user embedding")
parser.add_argument('--threshold_item', default= 1.0, help ="threshold for item embedding")

parser.add_argument('--attention', default='base',type =str, help="base: use self-attention fast_cluster: fast clustered attention ")
parser.add_argument('--SSE', default = False, type= str2bool, help="Stochastic Shared Embedding")


parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start trai.")
parser.add_argument("--seed", default=1, type=int)

#learning related
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")


parser.add_argument(
    "--training_data_ratio",
    default=0.5, #change to 1.0
    type=float,
    help="percentage of training samples used for training - robustness analysis",)
args = parser.parse_args()


#log
if not os.path.isdir('result_log'):
    os.makkedirs('result_log')
if not os.path.isdir('result_log/'+ args.dataset + '_' + args.train_dir):
    os.makedirs('result_log/' + args.dataset + '_' + args.train_dir)
with open(os.path.join('result_log/'+ args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()



if __name__ == '__main__':
    # dataset
    log_name = f"{args.dataset}_{args.train_dir}"
    args.log_file = os.path.join('result_log/',args.dataset + '_' + args.train_dir +  ".txt")
    checkpoint = log_name + ".pt"
    args.checkpoint_path = os.path.join('result_log', log_name, checkpoint)


    set_seed(args.seed)
    args.data_file = 'data/' + args.dataset + ".txt"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.train_matrix = valid_rating_matrix

    args.user_size = len(user_seq) + 1
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    cluster_dataset = SequentialDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)
    
    train_dataset = SequentialDataset(args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SequentialDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SequentialDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = UPTRec(args)
    trainer = UPTRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)
    
    print("Train UPTRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
    for epoch in range(args.num_epochs):
        trainer.train(epoch)

        #evaluate on NDCG@10
        scores, _ = trainer.valid(epoch, full_sort=True)
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")

        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

        print(log_name)
        print(result_info)
        with open(args.log_file, "a") as f:
            f.write(log_name + "\n")
            f.write(result_info + "\n")


    print("Main.py");import IPython; IPython.embed(colors='Linux');exit(1)
    # adjusted code ends here

    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    num_batch  = len(user_train)//args.batch_size

    f = open(os.path.join('result_log/'+args.dataset +'_' + args.train_dir,'log.txt'), 'w')
    sampler = WarpSampler(user_train, usernum, itemnum, SSE = args.SSE,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,
                          threshold_user = args.threshold_user, threshold_item = args.threshold_item)
    
    model = UPTRec(usernum, itemnum, args).to(args.device)
    

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    best_test_ndcg = 0.0
    best_test_hit = 0.0
    best_epoch = 0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
           
            start_time = time.time()

            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg, args)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_embedding.parameters(): loss += args.l2_emb * torch.norm(param)

            loss.backward()
            adam_optimizer.step()

            end_time = time.time()
            elapsed_time = end_time - start_time

            print("loss in epoch {} iteration {}: {} time: {}".format(epoch, step, loss.item(),elapsed_time)) # expected 0.4~0.6 after init few epochs
        if epoch == 1 or epoch == 2 or epoch % 5 == 0:# == 2 :
        #if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1

            #if current score exceeds
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)

            print('Evaluating', end='')
            
            current_valid_ndcg, current_valid_hit = t_valid[0], t_valid[1] 
            current_test_ndcg, current_test_hit = t_test[0], t_test[1]
            
            if best_test_hit > current_test_hit:
                best_test_hit = current_test_hit
                best_test_ndcg = current_test_ndcg
                best_epoch = epoch

            f.write('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) \n'
            % (epoch, T, args.k, current_valid_ndcg ,args.k, current_valid_hit
               ,args.k, current_test_ndcg,args.k, current_test_hit))

            f.flush()
            t0 = time.time()
            model.train()
    
        
        if epoch == args.num_epochs:
            f.write('finished\n')
            f.write('Best test results with HIT occured at epoch:%d, test (NDCG@%d: %.5f, HR@%d: %.5f) \n'
                    %(epoch, args.k, best_test_ndcg, args.k, best_test_hit))
            
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.item_hidden_units,args.user_hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join('result_log/'+folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
    
'''
python main.py --dataset=Beauty --train_dir=test
'''