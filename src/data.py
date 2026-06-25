import scanpy as sc
import ot
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import os
import numpy as np

class Load10xAdata:
    """
    Data loader for 10x Visium spatial transcriptomics data.
    
    This class handles loading, preprocessing, and spatial graph construction
    for 10x Visium spatial transcriptomics datasets. It supports various
    preprocessing steps including normalization, scaling, and spatial
    neighborhood graph construction.
    
    Args:
        path (str): Path to the 10x Visium data directory
        n_top_genes (int): Number of highly variable genes to select
        n_neighbors (int): Number of nearest neighbors for graph construction
        radius (int): Radius for spatial neighborhood definition
        label (bool): Whether to load ground truth labels
        filter_na (bool): Whether to filter out spots with missing labels
        epoch_num (int): Number of epochs for processing
    """
    def __init__(self, path: str, n_top_genes: int = 3000, n_neighbors: int = 6, radius: int = 150, label: bool = True, filter_na: bool = True):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.adata = None
        self.label = label
        self.filter_na = filter_na

    def load_data(self):
        """
        Load 10x Visium data including count matrix and spatial images.
        """
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()
            
    def load_label(self):
        """
        Load ground truth spatial domain labels from truth.txt file.

        Matches labels to spots by barcode to handle ordering differences
        between truth.txt and the h5 feature matrix.

        Optionally filters out spots with missing labels based on filter_na setting.
        """
        df_meta = pd.read_csv(os.path.join(self.path, 'truth.txt'), sep='\t', header=None)
        df_meta.columns = ['barcode', 'ground_truth']
        df_meta = df_meta.set_index('barcode')

        # Match labels to adata spots by barcode
        adata_barcodes = self.adata.obs_names
        self.adata.obs['ground_truth'] = adata_barcodes.map(df_meta['ground_truth']).values

        # Filter out spots with missing labels if requested
        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]

    def preprocess(self):
        """
        Standard preprocessing pipeline for spatial transcriptomics data.
        
        Steps include:
        1. Highly variable gene selection
        2. Total count normalization
        3. Log transformation
        4. Feature scaling
        """
        # Select highly variable genes
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
        
        # Normalize to 10,000 counts per spot
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(self.adata)
        
        # Scale features with clipping at 10
        sc.pp.scale(self.adata, zero_center=False, max_value=10)   

    def construct_interaction(self):
        """
        Construct spatial neighborhood graph based on Euclidean distance.
        
        Creates adjacency matrices for spatial neighborhoods using distance
        threshold and stores them in adata.obsm for later use in graph
        neural network training.
        """
        # Get spatial coordinates
        position = self.adata.obsm['spatial']
        
        # Calculate pairwise distances
        distance_matrix = ot.dist(position, position, metric='euclidean')
        
        # Create binary adjacency matrix based on distance threshold
        adj = (distance_matrix <= self.radius).astype(int)

        # Store neighborhood graph
        self.adata.obsm['graph_nei'] = adj

        n_spots = position.shape[0]
        top_k = self.n_neighbors+1
        sur_adj = np.zeros((n_spots, n_spots), dtype=int)

        for i in range(n_spots):
            distances = distance_matrix[i]
            nearest_indices = np.argsort(distances)[:top_k]
            sur_adj[i, nearest_indices] = 1

        self.adata.obsm['sur_adj'] = sur_adj

    def generate_gene_expr(self):
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['feat'] = feat


    def run(self):
        self.load_data()
        if self.label:
            self.load_label()
        self.preprocess()
        self.generate_gene_expr()
        self.construct_interaction()
      
        print('adata load done')
        return self.adata
    
class LoadAdata:
    def __init__(self, path: str, tech: str = 'visium', n_neighbors: int = 6, radius: int = 150, label: bool = True, filter_na: bool = True, n_top_genes: int = 3000, preprocessed: bool = False):
        self.path = path
        self.tech = tech
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.adata = None
        self.label = label
        self.filter_na = filter_na
        self.n_top_genes = n_top_genes
        self.preprocessed = preprocessed

    def load_data(self):
        self.adata = sc.read_h5ad(self.path)
        self.adata.var_names_make_unique()
            
    def load_label(self):
        if 'label' in self.adata.obs.columns and 'ground_truth' not in self.adata.obs.columns:
            self.adata.obs['ground_truth'] = self.adata.obs['label']

        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]

    def preprocess(self):
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)   

    def construct_interaction(self):
        
        position = self.adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')

        n_spots = position.shape[0]
        top_k = self.n_neighbors+1
        sur_adj = np.zeros((n_spots, n_spots), dtype=int)

        for i in range(n_spots):
            distances = distance_matrix[i]
            nearest_indices = np.argsort(distances)[:top_k]
            sur_adj[i, nearest_indices] = 1
        self.adata.obsm['sur_adj'] = sur_adj

        if self.tech == 'visium':
            adj = (distance_matrix <= self.radius).astype(int)
        else:
            adj = sur_adj
        self.adata.obsm['graph_nei'] = adj

        avg_deg = np.sum(adj) / n_spots
        print(f"Constructed adjacency matrix with average degree: {avg_deg:.2f}")

    def generate_gene_expr(self):
        use_hvg = ('highly_variable' in self.adata.var.columns) and self.adata.var['highly_variable'].any()
        if use_hvg:
            adata_vars = self.adata[:, self.adata.var['highly_variable']]
            Xsrc = adata_vars.X
        else:
            Xsrc = self.adata.X

        if isinstance(Xsrc, csc_matrix) or isinstance(Xsrc, csr_matrix):
            feat = Xsrc.toarray()[:, ]
        else:
            feat = Xsrc[:, ]

        self.adata.obsm['feat'] = feat

    def run(self):
        self.load_data()
        if self.label:
            self.load_label()
        if not self.preprocessed:
            self.preprocess()
        self.generate_gene_expr()
        self.construct_interaction()
      
        print('adata load done')
        return self.adata