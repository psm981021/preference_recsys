
import numpy as np
import torch
import time


def assign_clusters_cpu(hashes, lengths, centroids, group):
    N = hashes.size(0)
    H = hashes.size(1)
    L = hashes.size(2)
    K = centroids.size(2)

    for n in range(N):
        maxl = lengths[n]
        for l in range(maxl):
            for h in range(H):
                hash_val = hashes[n, h, l]
                min_distance = 1000
                assignment = K + 1
                for k in range(K):
                    distance = bin(hash_val ^ centroids[n, h, k]).count('1')
                    if distance < min_distance:
                        min_distance = distance
                        assignment = k
                group[n, h, l] = assignment

def kmeans_cpu(hashes, lengths, clusters, args):
    N = hashes.size(0)
    H = hashes.size(1)
    L = hashes.size(2)
    centroids = None
    counts = None
    group = None

    if group is None:
        group = torch.empty((N, H, L), dtype=torch.int32)
    if centroids is None:
        centroids = torch.empty((N, H, clusters), dtype=torch.int64)
        centroids[:, :, :] = hashes[:, :, torch.tensor(np.random.choice(L, size=args, replace=False))]
    K = centroids.shape[2]
    if counts is None:
        counts = torch.empty((N, H, K), dtype=torch.int32)
    
    assign_clusters_cpu(hashes, lengths, centroids, group)
    
    return group, counts



### cluster gpu

def hamming_distance(a, b):
    # Perform XOR operation between tensors
    xor_result = a ^ b
    
    # Count the number of set bits (i.e., 1s) in the result tensor
    hamming_dist = xor_result.to(torch.int).sum()

    return hamming_dist

# @torch.jit.script
def assign_clusters_kernel(hash_codes, lengths, centroids, labels, distances):
    N, H, L = hash_codes.shape
    K = centroids.shape[2]

    for n in range(N):
        for h in range(H):
            for l in range(L):
                if l < lengths[n]:
                    
                    x = hash_codes[n, h, l]
                    best_distance = 1000
                    best_cluster = -1
                    for k in range(K):
                        distance = bin(x ^ centroids[n, h, k]).count('1')
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = k
                    labels[n, h, l] = best_cluster
                    distances[n, h, l] = best_distance

def assign_clusters_kernel_works(hash_codes, lengths, centroids, labels, distances, n_blocks_per_sequence, MAX=65):
    """
    Assigns each hash code to its closest centroid based on Hamming distance
    It updates labels tensor with the assigned cluster indices and distances tensor
    """
    N, H, L = hash_codes.shape
    K = centroids.shape[2]

    sequence_indices = torch.arange(N * H, device=hash_codes.device)
    shared_means = centroids[sequence_indices // H, sequence_indices % H]

    for sequence_index in range(N * H):
        n = sequence_index // H
        h = sequence_index % H

        for blockIdx_x in range(n_blocks_per_sequence):
            for threadIdx_x in range(1024):
                l = (blockIdx_x * 1024) + threadIdx_x

                if l >= L:
                    continue

                if l >= lengths[n]:
                    labels[n, h, l] = K + 1
                    distances[n, h, l] = -1
                    continue

                x = hash_codes[n, h, l]
                best_distance = MAX
                best_cluster = 0
                for cluster in range(K):
                    
                    dist = hamming_distance(x, shared_means[cluster])
                    if dist < best_distance:
                        best_distance = dist
                        best_cluster = cluster

                labels[n, h, l] = best_cluster
                distances[n, h, l] = best_distance

def bit_count_kernel(labels, hash_codes, counts, cluster_bit_counts):
    N, H, L = labels.shape
    K, B = cluster_bit_counts.shape[2:]

    for n in range(N):
        for h in range(H):
            for l in range(L):
                if labels[n, h, l] < K:  # Skip if label corresponds to an empty cluster
                    x = hash_codes[n, h, l]
                    for i in range(B):
                        bit = 1 << i
                        if x & bit:
                            val_to_add = 1
                        else:
                            val_to_add = -1
                        cluster_bit_counts[n, h, labels[n, h, l], i] += val_to_add
                    counts[n, h, labels[n, h, l]] += 1

def bit_count_kernel_works(labels, hash_codes, counts, cluster_bit_counts):
    """

    Computes bit counts for each cluster based on the assigned labelsand has codes
    Updates counts and cluster_bit_counts 

    """
    N, H, L = labels.shape
    K, B = cluster_bit_counts.shape[2:]

    for full_idx in range(L * N * H):
        sequence_index = full_idx // L
        n = sequence_index // H
        h = sequence_index % H
        l = full_idx - n * (H * L) - h * L

        if n >= N:
            continue

        x = hash_codes[n, h, l]
        val_to_add = -1
        best_cluster = labels[n, h, l]

        if best_cluster == (K + 1):
            continue

        for i in range(B):
            bit = 1 << i
            if (x & bit) > 0:
                val_to_add = 1
            else:
                val_to_add = -1
            cluster_bit_counts[n, h, best_cluster, i] += val_to_add
        counts[n, h, best_cluster] += 1

        
def compute_means_kernel(counts, cluster_bit_counts, centroids):
    N, H, K = counts.shape
    B = cluster_bit_counts.shape[3]

    for n in range(N):
        for h in range(H):
            for k in range(K):
                mean_k = 0
                for i in range(B):
                    if cluster_bit_counts[n, h, k, i] > 0:
                        mean_k |= (1 << i)
                centroids[n, h, k] = mean_k


def compute_means_kernel_works(counts, cluster_bit_counts, centroids):
    """
    
    Computes the new centroids based on the assigned labels and bit counts
    Updates centroids tensor
    
    """
    N, H, K = counts.shape
    B = cluster_bit_counts.shape[3]

    for full_idx in range(K * N * H):
        sequence_idx = full_idx // K
        n = sequence_idx // H
        h = sequence_idx % H
        k = full_idx % K

        if full_idx >= (K * N * H):
            continue

        mean_k = 0
        MAX = 1 << B

        if counts[n, h, k] == 0:
            centroids[n, h, k] = torch.randint(0, MAX, (1,), device=centroids.device)
            continue

        for i in range(B):
            if cluster_bit_counts[n, h, k, i] == 0:
                cluster_bit_counts[n, h, k, i] = torch.randint(0, 2, (1,), device=centroids.device)
            if cluster_bit_counts[n, h, k, i] > 0:
                mean_k |= (1 << i)
        centroids[n, h, k] = mean_k

def kmeans_gpu(hash_codes, lengths, centroids, distances, cluster_bit_counts, labels, counts, iterations):
    N, H, L = hash_codes.shape
    K, B = cluster_bit_counts.shape[2:]

    for itr in range(iterations):
        #print(f"Iteration: {itr + 1}")
        start_time = time.time()

        assign_clusters_kernel(hash_codes, lengths, centroids, labels, distances) #,(L - 1) // 1024 + 1)
        counts.zero_()
        cluster_bit_counts.zero_()
        bit_count_kernel(labels, hash_codes, counts, cluster_bit_counts)
        compute_means_kernel(counts, cluster_bit_counts, centroids)

        end_time = time.time()
        elapsed_time = end_time - start_time

        #print(f"Elapsed time for iteration {itr + 1}: {elapsed_time} seconds")
    
    #return centroids


def compute_hashes(X, A, H=None):

    N, D = X.size()
    B, _ = A.size()
    assert D + 1 == A.size(1), "Bias expected for the parameters"
    
    #initialize Hash
    H = torch.zeros(N, dtype=torch.int64, device =X.device)

    # Compute dot product of X with each plane in A
    dot_products = torch.matmul(X, A[:, :-1].t()) + A[:, -1]
    for i in range(B):
        bit = dot_products[:, i] > 0
        
        H = torch.bitwise_or(H, bit << i)
    
    return H
    
def clustered_aggregate(X, G, F, lengths, Y=None):
    # X: N H L E : input vectors
    # G: Group indices of tensor N H L, specify group for each input vector
    # F: Factors tensor of shape N H C, factors to multiply each input vector within groups
    # lengths: Tensor of length N, length of each sequence in batch
    # Y: represents aggregated values for each clusters, used as query in attention mechanism N C E
    N, H, L, E = X.size()
    C = F.size(2)
    Y = torch.zeros(N, H, C, E, device=X.device, dtype=X.dtype)

    # Better computation !!
    for n in range(N): # batch
        for h in range(H): # head
            for l in range(L): # each position
                k_cur = G[n][h][l] # retrieve current group index
                f_cur = F[n][h][k_cur] # retrieve factor from current group
                for e in range(E):
                    Y[n][h][k_cur][e] += f_cur * X[n][h][l][e]

    # This code works!! returns shape of N x H x C x E
    # for n in range(N): # batch
    #     for h in range(H): # head
    #         for l in range(L): # each position
    #             k_cur = G[n][h][l] # retrieve current group index
    #             f_cur = F[n][h][k_cur] # retrive factor from current group
    #             y_cur = Y[n][h][k_cur]
    #             for e in range(E):
    #                 y_cur += f_cur * X[n][h][l][e].item()  # update aggregated vector

    return Y

def set_group(C, E):
    """
    Divide the cluster into groups of equal size
    as constrained by the shared memory"""
    C_per_block = int(192 * 64 / (E+1))
    G_min = (C + C_per_block - 1) // C_per_block
    for G in range(G_min, C+1):
        if C % G == 0:
            return G

def clustered_broadcast(Y, groups, counts, factors, X=None):
    # Y: Aggregated tensor N C E
    device = Y.device

    if X is None:
        X = torch.zeros(
            groups.shape + (Y.shape[-1],),
            device=device,
            dtype=Y.dtype
        )
    N, H, C, E = Y.shape
    _, _, L, _ = X.shape

    # threads = 256
    # G = set_group(C, E)
    # group_counts = counts.view(N, H, G, -1).sum(-1)
    # block_counts = (group_counts + threads - 1) // threads
    # total_blocks = block_counts.sum().item()

    # indx_maps = torch.ones(
    #     (total_blocks, 5),
    #     device=X.device,
    #     dtype=torch.int32
    # )

    # Iterate over each group and perform broadcast operation
    for n in range(N):
        for h in range(H):
            # Get group indices for current batch and head
            group_indices = groups[n, h] 

            # Get group factors for current batch and head
            group_factors = factors[n, h]  

            # Get count of clusters in the group
            group_count = counts[n, h]  

            for l in range(C):
                # Iterate over each query in the sequence
                # Get query index for current sequence and position
                query_index = group_indices[l]  

                if query_index < group_count[l]:
                    # Get factor for current query
                    factor = group_factors[query_index]  
                    for c in range(C):
                        # Iterate over each cluster in the group
                        for e in range(E):
                            # Iterate over each element in the vector
                            X[n, h, l, e] = Y[n, h, c, e] * factor

    return X

    indx_maps = create_maps(group_counts, block_counts) #provides indices to iterate over each group in the batch

    for idx in indx_maps:
        n = idx[0]
        h = idx[1]
        g = idx[2]
        q_id = idx[3]
        n_queries = idx[4]
        clusters_to_load = C // G.size(2) 
        cluster_offset = g * clusters_to_load
        shared_values = Y[n][h][cluster_offset:cluster_offset + clusters_to_load] #from current group
        shared_factors = F[n][h][cluster_offset:cluster_offset + clusters_to_load] #from current group
        for l in range(q_id, q_id + n_queries):
            k = G[n][h][l]
            k -= cluster_offset
            factor = shared_factors[k]
            for e in range(E):
                # No explicit return as 'X' is updated in place
                X[n][h][l][e] = shared_values[k][e] * factor
                

def create_maps(group_counts, block_counts):
    # group_counts: Tensor of shape N H G number of vectors per group
    # blcok_counts: Tensor of shape N H G number of blocks per group


    maps = []
    N, H, G = group_counts.size()
    for n in range(N): # batch
        for h in range(H): # head
            acc_g_count = 0
            for g in range(G): # group
                g_count = group_counts[n][h][g] #calculate total number of vectors
                blocks = block_counts[n][h][g] #calculate total number of blocks
                for b in range(blocks):
                    maps.append([n, h, g, acc_g_count, g_count])
                    acc_g_count += g_count
    return maps # return generated maps 



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