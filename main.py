import argparse
import os
import time
import torch
from utils import data_partition, WarpSampler, evaluate, evaluate_valid, early_stopping
from models import UPTRec
import numpy as np

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
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
#parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--item_hidden_units', default= 50, type=int, help="hidden units for item embedding")
parser.add_argument('--user_hidden_units', default= 50, help ="hidden units for user embedding")
parser.add_argument('--cluster_num', default =10, help ="number of clusters")
parser.add_argument('--threshold_user', default= 1.0, help ="threshold for user embedding")
parser.add_argument('--threshold_item', default= 1.0, help ="threshold for item embedding")
parser.add_argument('--attention_mask', default='base',type=str,help="base, cluster")
parser.add_argument('--SSE', default = False, type= str2bool, help="Stochastic Shared Embedding")
parser.add_argument('--k', default = 10, type=ndcg_k_type , help ="Metrics@K")
parser.add_argument('--early_stopping', default = True, type = bool, help ="enable early stopping")
parser.add_argument('--patience', default= 20, type= int, help="Number of epochs with no improvement after training will be stopped")


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

    best_valid_ndcg = 0.0
    best_valid_hit = 0.0

    best_test_ndcg = 0.0
    best_test_hit = 0.0

    cur_step = 1
    update_epoch = 0
    
    flag_early_stopping = False

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

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

            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        #if epoch == 1 :
        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1

            #if current score exceeds
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)

            print('Evaluating', end='')
            
            current_valid_ndcg, current_valid_hit = t_valid[0], t_valid[1] 
            current_test_ndcg, current_test_hit = t_test[0], t_test[1]


            # HIT@K will be the criteria
            if current_valid_hit > best_valid_hit:
                cur_step = 1
                best_valid_hit, best_valid_ndcg = current_valid_hit, current_valid_ndcg
                best_test_hit, best_test_ndcg = current_test_hit, current_test_ndcg
            
            
            # if evaluation not improved
            elif current_valid_hit < best_valid_hit or current_valid_ndcg < best_valid_ndcg:
                cur_step+=1

                if args.early_stopping == True and cur_step > args.patience:
                    flag_early_stopping = True
                    f.write('(Best Results HIT) Early stopping occurred on epoch: %d, time: %.4f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f)\n'
                    % (epoch, T, args.k, best_valid_ndcg, args.k, best_valid_hit,
                        args.k, best_test_ndcg, args.k, best_test_hit))

        
            f.write('cur_step:%d, epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) \n'
            % (cur_step, epoch, T, args.k, current_valid_ndcg ,args.k, current_valid_hit
               ,args.k, current_test_ndcg,args.k, current_test_hit))

            f.flush()
            t0 = time.time()
            model.train()
    
        # 수정
        if epoch == args.num_epochs and flag_early_stopping == True:
            f.write('finished\n')
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
    
def convert_seconds(seconds):
    days = seconds // (24 * 3600)
    remaining_seconds = seconds % (24 * 3600)
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    minutes = remaining_seconds // 60
    remaining_seconds %= 60

    return days, hours, minutes, remaining_seconds


seconds = int(input())

days, hours, minutes, remaining_seconds = convert_seconds(seconds)

print(f"{days} days {hours:02d}:{minutes:02d}:{remaining_seconds:02d}")