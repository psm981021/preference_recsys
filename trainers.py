import torch

def kmeans_simple_gpu(hashes, clusters, iterations=10):
    """
    Perform K-means clustering on hash codes using GPU.

    Args:
    - hashes (torch.Tensor): Hash codes tensor of shape (N, L).
    - clusters (int): Number of clusters.
    - iterations (int): Number of iterations.

    Returns:
    - group (torch.Tensor): Group indices tensor of shape (N, L).
    - counts (torch.Tensor): Cluster counts tensor of shape (N, clusters).
    """
    device = hashes.device
    N, L = hashes.shape
    K = clusters

    group = torch.empty((N, L), dtype=torch.int32, device=device)
    counts = torch.empty((N, K), dtype=torch.int32, device=device)
    centroids = hashes[:, torch.randint(0, L, (K,), device=device)]

    for _ in range(iterations):
        assign_clusters_kernel(hashes, centroids, group)
        counts.zero_()
        bit_count_kernel(group, hashes, counts)
        compute_means_kernel(counts, centroids)

    return group, counts

@torch.jit.script
def assign_clusters_kernel(hash_codes, centroids, labels):
    N, L = hash_codes.shape
    K = centroids.shape[0]

    for n in range(N):
        for l in range(L):
            x = hash_codes[n, l]
            best_distance = float('inf')
            best_cluster = -1
            for k in range(K):
                distance = (x ^ centroids[k]).to(torch.int).sum()
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = k
            labels[n, l] = best_cluster

@torch.jit.script
def bit_count_kernel(labels, hash_codes, counts):
    N, L = labels.shape
    K = counts.shape[1]
    B = hash_codes.shape[1]  # Assuming hash codes have shape (N, B)

    for n in range(N):
        for l in range(L):
            if labels[n, l] < K:
                x = hash_codes[n, l]
                for i in range(B):
                    bit = 1 << i
                    if x & bit:
                        val_to_add = 1
                    else:
                        val_to_add = -1
                    counts[n, labels[n, l]] += val_to_add

@torch.jit.script
def compute_means_kernel(counts, centroids):
    N, K = counts.shape

    for n in range(N):
        for k in range(K):
            mean_k = 0
            centroids[k] = mean_k
