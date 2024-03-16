
import numpy as np
import torch

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
    return (a ^ b).popcount()

def assign_clusters_kernel(hash_codes, lengths, centroids, labels, distances, n_blocks_per_sequence, MAX=65):
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
        assign_clusters_kernel(hash_codes, lengths, centroids, labels, distances, (L - 1) // 1024 + 1)
        counts.zero_()
        cluster_bit_counts.zero_()
        bit_count_kernel(labels, hash_codes, counts, cluster_bit_counts)
        compute_means_kernel(counts, cluster_bit_counts, centroids)