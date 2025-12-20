import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
import os
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def adj_to_edge_index_with_weight(adj):
    """
    Convert adjacency matrix to edge index and edge weight format for PyTorch Geometric.
    
    Args:
        adj (torch.Tensor): Adjacency matrix [num_nodes, num_nodes]
        
    Returns:
        tuple: (edge_index, edge_weight) where edge_index is [2, num_edges] 
               and edge_weight is [num_edges]
    """
    indices = adj.nonzero(as_tuple=False)  # Get non-zero indices
    edge_index = indices.T                 # Transpose to get [2, num_edges] format
    edge_weight = adj[indices[:, 0], indices[:, 1]]  # Extract corresponding weights
    return edge_index, edge_weight

def _nan2inf(x):
    """
    Replace NaN values with infinity for numerical stability in loss computations.
    
    Args:
        x (torch.Tensor): Input tensor potentially containing NaN values
        
    Returns:
        torch.Tensor: Tensor with NaN values replaced by infinity
    """
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='norm_emb', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def kmeans(adata, num_cluster, used_obsm='norm_emb', random_seed=2023):

    np.random.seed(random_seed)
    a = adata.obsm[used_obsm]
    kmeans = KMeans(n_clusters=num_cluster, random_state=random_seed)
    kmeans.fit(a)
    adata.obs['kmeans'] = pd.Categorical(kmeans.labels_)

    return adata

def clustering(adata, used_obsm='emb', n_clusters=7, radius=50, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False, percentage=0.8):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=used_obsm, num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(radius,adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(radius,adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain'] 
    elif method == 'kmeans':
        adata = kmeans(adata, used_obsm=used_obsm, num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['kmeans']

    if refinement:
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


    
def search_res(radius,adata, n_clusters, method='leiden', use_rep='norm_emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    ress=[]
    best=None
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    sc.tl.leiden(adata, random_state=0, resolution=end)
    count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique > n_clusters +2):
        print(count_unique)
        print('too large, continue adjusting')
        end = end - 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique < n_clusters + 2):
        print(count_unique)
        print('too small, continue adjusting')
        end = end + 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())


    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters:
            print('calculate metric ARI')
            # calculate metric ARI
            new_type = refine_label(adata, radius, key='leiden')
            adata.obs['leiden'] = new_type

            ARI = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
            adata.uns['ARI'] = ARI
            ress.append((res,ARI))
            print('ARI:', ARI)

        if count_unique == n_clusters-2:
            label = 1
            best = max(ress, key=lambda x: x[1])
            print(best)
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return best[0]