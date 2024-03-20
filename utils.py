
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fast_cluster import *
from fast_cluster import kmeans_cpu, kmeans_gpu

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum,SSE, batch_size, maxlen, result_queue, SEED,
                    threshold_user, threshold_item):
    def sample():
        
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        
        seq = np.zeros([int(maxlen)], dtype=np.int32)
        pos = np.zeros([int(maxlen)], dtype=np.int32)
        neg = np.zeros([int(maxlen)], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = int(maxlen) - 1

        ts = set(user_train[user])

        for i in reversed(user_train[user][:-1]):
            if SSE == True:
                # SSE for user side (2 lines)
                if random.random() > threshold_item:
                    i = np.random.randint(1, itemnum + 1)
                    nxt = np.random.randint(1, itemnum + 1)

            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        if SSE == True:
            # SSE for item side (2 lines)
            if random.random() > threshold_user:
                user = np.random.randint(1, usernum + 1)
            # equivalent to hard parameter sharing
            #user = 1

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

#WarpSampler needs to be changed to dataloader
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum,SSE, batch_size=64, maxlen=10, n_workers=1,
                threshold_user =1.0, threshold_item = 1.0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      SSE,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      threshold_user,
                                                      threshold_item
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()



# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    #not a seq version
    f = open('data/%s.txt' %fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')

        u = int(u)
        i = int(i)

        usernum = max(u, usernum)
        itemnum = max(i,itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


## check
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        #
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]],args) # 1 x item_idx
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item() # item_idx

        valid_user += 1

        if rank < args.k:
        #if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        
        

    return NDCG / valid_user, HT / valid_user

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        #predictions = -model.predict(np.array([u]), np.array([seq]), np.array(item_idx), args)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]],args)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        
        if rank < args.k:
        #if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def early_stopping(value, best, cur_step, max_step, bigger=True):
    """
    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
            best result after this step
        - int,
            the number of consecutive steps that did not exceed the best result after this step
        - bool,
            whether to stop
        - bool,
            whether to update
    """

    stop_flag = False
    update_flag = False

    if bigger:
        if value >= best:
            cur_step =0
            best = value
            update_flag = True
        else:
            cur_step+=1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best: 
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    
    return best, cur_step, stop_flag, update_flag 




def cluster(
        hashes,
        lengths,
        args,
        group = None,
        counts = None,
        centroids = None,
        distances = None, 
        bitcounts = None,
        iterations = 10,
        bits =32
):
    """
    original code from https://github.com/idiap/fast-transformers/tree/master/fast_transformers/clustering
    Cluster using hashing of Kmeans using hamming distance

    Arguments
    ---------
        hashes: A long tensor of shape (B, T, C) containing a hashcode for each
                query.
        lengths: An int tensor of shape (B,) containing the sequence length for
                 each sequence in hashes.
        groups: An int tensor buffer of shape (B, T, C) contaning the cluster
                in which the corresponding hash belongs to.
        counts: An int tensor buffer of shape (B, H, K) containing the number
                of elements in each cluster.
        centroids: A long tensor buffer of shape (B, H, K) containing the
                   centroid for each cluster.
        distances: An int tensor of shape (B, H, T) containing the distance to
                   the closest centroid for each hash.
        bitcounts: An int tensor of shape (B, H, K, bits) containing the number
                   of elements that have 1 for a given bit.
        clusters: The number of clusters to use for each sequence. It is
                  ignored if centroids is not None.
        iterations: How many k-means iterations to perform.
        bits: How many of the least-significant bits in hashes to consider.

    Returns
    -------
        groups and counts as defined above.
    """

    device = hashes.device

    N, H, L = hashes.shape
    clusters = args.cluster_num

    if device.type == "cpu":
        if group is None:
            group = torch.empty((N, H, L), dtype=torch.int32)
        if centroids is None:
            centroids = torch.empty((N, H, clusters), dtype=torch.int64)
            centroids[:, :, :] = hashes[:, :, torch.randint(0, L, (clusters,))]
        K = centroids.shape[2]
        if counts is None:
            counts = torch.empty((N, H, K), dtype=torch.int32)

        group, counts = kmeans_cpu(hashes, lengths, clusters, args)

        return group, counts
    else:
        if group is None:
            group = torch.empty((N, H, L), dtype=torch.int32, device=device)
        if centroids is None:
            centroids = hashes[:, :, torch.randint(0, L, (clusters,), device=device)]
        
        if counts is None:
            counts = torch.empty((N, H, clusters), dtype=torch.int32, device=device)
        if distances is None:
            distances = torch.empty((N, H, L), dtype=torch.int32, device=device)
        if bitcounts is None:
            bitcounts = torch.empty((N, H, clusters, bits), dtype=torch.int32, device=device)

        kmeans_gpu(
            hashes,
            lengths,
            centroids,
            distances,
            bitcounts,
            group,
            counts,
            iterations)
        
    
    return group, counts