import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import datetime
import tqdm
from .net import Model_GCN, ZINB
from .utils import adj_to_edge_index_with_weight
from sklearn import metrics
import scanpy as sc
import matplotlib.pyplot as plt
import datetime
import os
import gc

def regularization(z, graph_neigh):
    """
    Tissue-level contrastive loss that encourages connected nodes to have similar embeddings
    while pushing unconnected nodes apart.
    
    Args:
        z: Node embeddings [N, D]
        graph_neigh: Adjacency matrix [N, N]
    
    Returns:
        Regularization loss value
    """
    # Normalize embeddings to unit vectors
    z_norm = F.normalize(z, p=2, dim=1)
    # Compute pairwise cosine similarity matrix
    sim_mat = torch.mm(z_norm, z_norm.t())
    # Create edge mask for existing connections
    edge_mask = graph_neigh > 0

    # Calculate average similarity for connected node pairs
    edge_sim = torch.mul(sim_mat, torch.mul(graph_neigh, edge_mask)).sum()
    edge_count = edge_mask.sum().float() + 1e-6
    
    # Calculate average similarity for unconnected node pairs
    non_edge_mask = ~edge_mask
    non_edge_sim = torch.mul(sim_mat, non_edge_mask).sum()
    non_edge_count = non_edge_mask.sum().float() + 1e-6
    
    # Maximize edge similarity and minimize non-edge similarity
    loss = -edge_sim/edge_count + non_edge_sim/non_edge_count
    return loss

class spaDGC:
    def __init__(self, args, config, adata):
        """
        Initialize SpaDGC model with configuration parameters.
        
        Args:
            args: Command line arguments
            config: Configuration dictionary containing hyperparameters
            adata: AnnData object containing spatial transcriptomics data
        """
        self.args = args
        self.config = config
        self.adata = adata
        self.seed = config.get('seed', 3407)
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.dim_hidden = config['dim_hidden']
        self.dim_out = config['dim_out']
        self.num_epochs = config['num_epochs']
        self.num_classes = config['num_classes']
        self.num_gene = config['num_gene']
        self.log_dir = config.get('log_dir', 'logs')
        self.bar_format = '{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training strategy parameters
        self.warmup_epochs = config.get('warmup_epochs', 50)  # Epochs before graph updating starts
        self.update_interval = config.get('update_interval', 10)  # Graph update frequency
        self.num_dynamic = config.get('num_dynamic', 2)  # Number of dynamic edges to add per boundary node
        self.memory_factor = config.get('memory_factor', 0)  # Memory factor for graph smoothing
        self.verbose_interval = config.get('verbose_interval', 10)  # Logging frequency
        
        # Loss function weights
        self.alpha = config.get('alpha', 0.1)  # Neighborhood contrastive loss weight
        self.beta = config.get('beta', 0.1)   # Tissue-level contrastive loss weight
        
        # Initialize model and loss function
        self.model = Model_GCN(self.num_gene, self.dim_hidden, self.dim_out).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()  # Binary cross-entropy for contrastive learning
    
    def train(self):
        """
        Main training function for the SpaDGC model.
        
        This function implements the complete training pipeline including:
        - Data preparation and initialization
        - Iterative model training with dynamic graph updating
        - Convergence monitoring and evaluation
        - Final clustering and metrics calculation
        """
        
        # Validate input data
        if self.adata is None:
            raise ValueError("adata not load!")
        
        # Get number of spatial spots
        n_spot = self.adata.n_obs
        
        # Create labels for contrastive learning
        # one_matrix: positive pairs, zero_matrix: negative pairs
        one_matrix = torch.ones([n_spot, 1], dtype=torch.float32, device=self.device)
        zero_matrix = torch.zeros([n_spot, 1], dtype=torch.float32, device=self.device)
        label_CSL = torch.cat([one_matrix, zero_matrix], dim=1)

        # Load preprocessed features and adjacency matrices
        features_matrix = torch.tensor(self.adata.obsm['feat'], dtype=torch.float32, device=self.device)
        sur_adj = torch.tensor(self.adata.obsm['sur_adj'], dtype=torch.float32, device=self.device)
        
        # Extract spatial neighborhood edges from initial spatial adjacency matrix
        sur_edge_indices = (sur_adj > 0).nonzero(as_tuple=True)
        sur_i_indices, sur_j_indices = sur_edge_indices
        
        # Initialize graph neighborhood structure
        graph_neigh_init = torch.tensor(self.adata.obsm['graph_nei'], dtype=torch.float32, device=self.device)
        
        # Build neighbors dictionary for efficient neighbor lookup
        neighbors_dict = {}
        for i in range(n_spot):
            neighbors_dict[i] = (graph_neigh_init[i] > 0).nonzero().flatten().cpu().numpy()
        
        # Extract edge indices from initial graph
        edge_indices = (graph_neigh_init > 0).nonzero(as_tuple=True)
        i_indices, j_indices = edge_indices
        
        # Create working copy of the graph
        graph_neigh = graph_neigh_init.clone()
        edge_index, edge_weight = adj_to_edge_index_with_weight(graph_neigh)

        print('=== Training Start ===')
        
        # Initialize loss tracking lists for monitoring training progress
        loss_all, loss_ZINB_all, loss_tissue_all, loss_neigh_all = [], [], [], []


        # Setup optimizer with Adam algorithm
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Main training loop
        for epoch in tqdm.tqdm(range(self.num_epochs), bar_format=self.bar_format):

            # Set model to training mode
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            # Generate corrupted features for neighborhood-level contrastive learning
            # by randomly permuting the original features
            perm = torch.randperm(features_matrix.size(0))
            features_corrupted = features_matrix[perm]
            
            # z: node embeddings, pi/disp/mean: ZINB parameters, ret/ret_a: contrastive outputs
            z, pi, disp, mean, ret, ret_a = self.model(features_matrix, features_corrupted, edge_index, graph_neigh, edge_weight)
            
            # Calculate ZINB reconstruction loss for gene expression modeling
            loss_ZINB = ZINB(pi, theta=disp, ridge_lambda=0).loss(features_matrix, mean, mean=True)

            # Apply different loss strategies based on training phase
            if epoch < self.warmup_epochs:
                # Warmup phase: only use neighborhood-level contrastive learning, no tissue-level contrastive learning
                loss_tissue = torch.zeros_like(loss_ZINB)
                loss_neigh = (self.loss_CSL(ret, label_CSL) + self.loss_CSL(ret_a, label_CSL)) * self.alpha
            else:
                # Main training phase: add tissue-level contrastive learning
                loss_neigh = (self.loss_CSL(ret, label_CSL) + self.loss_CSL(ret_a, label_CSL)) * self.alpha
                loss_tissue = regularization(z, graph_neigh) * self.beta

            # Combine all loss components
            loss = loss_ZINB + loss_tissue + loss_neigh 
            
            # Record losses for monitoring and visualization
            loss_all.append(loss.item())
            loss_ZINB_all.append(loss_ZINB.item())
            loss_tissue_all.append(loss_tissue.item())
            loss_neigh_all.append(loss_neigh.item())

            # Backward pass and parameter update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping for stability
            optimizer.step()
            
            # Extract current embeddings for graph updating
            emb = z.detach().cpu().numpy()

            # Dynamic graph updating phase (after warmup period)
            if epoch >= self.warmup_epochs and epoch % self.update_interval == 0:
                with torch.no_grad():
                    # Store current graph for comparison
                    old_graph_neigh = graph_neigh.clone()

                    # Get current embeddings and perform clustering
                    emb = z.detach().cpu().numpy()
                    kmeans = KMeans(n_clusters=self.num_classes, n_init='auto', random_state=self.seed).fit(emb)
                    idx = kmeans.labels_
                    idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
                    
                    # Identify boundary candidate spots (spots probably at domain boundary)
                    # Boundary spots are those connected to spots in different clusters (according to inital spatial graph and temporary cluster labels)
                    is_boundary = torch.zeros(n_spot, dtype=torch.bool, device=self.device)
                    cross_cluster_mask = idx_tensor[sur_i_indices] != idx_tensor[sur_j_indices]
                    is_boundary[sur_i_indices[cross_cluster_mask]] = True
                    is_boundary[sur_j_indices[cross_cluster_mask]] = True

                    boundary_count = is_boundary.sum().item()
                    print(f"[Epoch {epoch}]: Found {boundary_count} boundary nodes ({boundary_count/n_spot:.2%})")

                    # Skip graph update if no boundary spots found
                    if boundary_count == 0:
                        continue

                    # Initialize new graph with original structure (initial spatial graph)
                    new_graph_neigh = graph_neigh_init.clone()

                    # Update edges between boundary spots based on embedding similarity
                    update_mask = is_boundary[i_indices] & is_boundary[j_indices]
                    update_i_indices = i_indices[update_mask]
                    update_j_indices = j_indices[update_mask]
                    
                    # Update edge weights based on embedding similarity if boundary edges exist
                    if len(update_i_indices) > 0:
                        # Normalize embeddings for cosine similarity calculation
                        z_norm = F.normalize(z, p=2, dim=1)
                        similarities = torch.sum(z_norm[update_i_indices] * z_norm[update_j_indices], dim=1)

                        # Separate same-cluster and cross-cluster edges for different treatments
                        is_same_cluster = idx_tensor[update_i_indices] == idx_tensor[update_j_indices]
                        same_cluster_edges = torch.where(is_same_cluster)[0]

                        # Strengthen edges within the same cluster
                        if len(same_cluster_edges) > 0:
                            i_same = update_i_indices[same_cluster_edges]
                            j_same = update_j_indices[same_cluster_edges]
                            sim_same = similarities[same_cluster_edges]

                            # Apply sigmoid activation to enhance strong similarities
                            weight_same = torch.sigmoid(sim_same * 5)
                            new_graph_neigh[i_same, j_same] = weight_same

                        # Weaken edges across different clusters
                        diff_cluster_edges = torch.where(~is_same_cluster)[0]
                        if len(diff_cluster_edges) > 0:
                            i_diff = update_i_indices[diff_cluster_edges]
                            j_diff = update_j_indices[diff_cluster_edges]
                            sim_diff = similarities[diff_cluster_edges]
                            # Reduce cross-cluster edge weights
                            weight_diff = (sim_diff+1)/2 * 0.5
                            new_graph_neigh[i_diff, j_diff] = weight_diff
                        
                        # Add dynamic edges to connect top-k similar spots within same clusters
                        if self.num_dynamic > 0:
                            current_dynamic = max(1, self.num_dynamic)
                            boundary_idx = torch.where(is_boundary)[0]
                            edges_added = 0

                            # For each boundary spot, find and connect to similar spots in same cluster
                            for i in boundary_idx:
                                i_val = i.item()
                                current_label = idx_tensor[i_val]

                                # Get current neighbors and find same-cluster candidates
                                neighbors = torch.tensor(neighbors_dict[i_val], device=self.device)
                                same_cluster_nodes = torch.where(idx_tensor == current_label)[0]
                                
                                # Exclude already connected neighbors and self
                                exclude_set = set(neighbors.cpu().numpy().tolist() + [i_val])
                                candidates = torch.tensor([n for n in same_cluster_nodes.cpu().numpy() if n not in exclude_set], 
                                  device=self.device)
                                
                                # Select most similar candidates for connection
                                if len(candidates) > 0:
                                    # Calculate similarity with candidates
                                    sim = torch.mm(z_norm[i:i+1], z_norm[candidates].t()).squeeze(0)
                                    if sim.dim() == 0:
                                        sim = sim.unsqueeze(0)
                                    
                                    # Adaptively select number of connections
                                    adaptive_k = min(current_dynamic, len(candidates))
                                    if adaptive_k > 0:
                                        # Select top-k most similar candidates
                                        _, top_indices = torch.topk(sim, adaptive_k)
                                        selected = candidates[top_indices]

                                        # Set edge weights based on similarity
                                        weights = torch.sigmoid(sim[top_indices] * 5)
                                        new_graph_neigh[i, selected] = weights
                                        new_graph_neigh[selected, i] = weights
                                        edges_added += adaptive_k

                    # Ensure graph symmetry by averaging with transpose
                    new_graph_neigh = (new_graph_neigh + new_graph_neigh.t()) / 2

                    # Update working graph
                    graph_neigh = new_graph_neigh.clone()
                    edge_index, edge_weight = adj_to_edge_index_with_weight(graph_neigh)

                    # Clean up memory
                    del new_graph_neigh, old_graph_neigh

            # Periodic memory cleanup to prevent memory accumulation
            if epoch % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Periodic progress logging
            if epoch % self.verbose_interval == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | ZINB: {loss_ZINB.item():.4f} | tissue: {loss_tissue.item():.4f} | neighborhood: {loss_neigh.item():.4f}")
        print("=== Training Completed ===")

        # Save trained model
        torch.save(self.model.state_dict(), 'model.pt')
        
        # Final clustering and evaluation
        emb = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_classes, n_init='auto', random_state=self.seed).fit(emb)
        idx = kmeans.labels_
        
        # Calculate clustering metrics
        ari_res = metrics.adjusted_rand_score(self.adata.obs['ground_truth'], idx)
        nmi_res = metrics.normalized_mutual_info_score(self.adata.obs['ground_truth'], idx)
        feat_rec = mean.detach().cpu().numpy()

        # Store results in AnnData object
        self.adata.obs['cluster'] = idx
        self.adata.obs['cluster'] = self.adata.obs['cluster'].astype('category')
        self.adata.uns['ari'] = ari_res
        self.adata.uns['nmi'] = nmi_res
        self.adata.obsm['emb'] = emb
        self.adata.obsm['X_rec'] = feat_rec
        print(f"Final clustering metrics: ARI_{ari_res:.3f}  NMI_{nmi_res:.3f}")

        # Generate and save training loss curves
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.log_dir, f"loss_curve_{timestamp}.png")
        plt.figure(figsize=(8, 5))
        plt.plot(loss_all, label='Total Loss', linewidth=2)
        plt.plot(loss_ZINB_all, label='ZINB Loss', linestyle='--')
        plt.plot(loss_tissue_all, label='Tissue Loss', linestyle='--')
        plt.plot(loss_neigh_all, label='Neighborhood Loss', linestyle='--')
        plt.axvline(self.warmup_epochs, color='gray', linestyle='--', label='Warmup End')
        plt.title("Training Losses over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss curve saved as {plot_path}")

    def draw_spatial(self, size=1.6, p=''):
        """
        Visualize spatial clusters on tissue image.
        
        Args:
            size (float): Size of the spots in the plot
            p (str): Prefix for the saved filename
        """
        title1 =  ["ground_truth", f"{self.args.slide}: ARI={self.adata.uns['ari']:.2f}, NMI={self.adata.uns['nmi']:.2f}"]
        sc.pl.spatial(self.adata,
                      img_key='hires',
                      size=size,
                      color=['ground_truth',"cluster"],
                      title = title1,
                      show=True, save=p+str(self.args.slide)+'.png')

    def draw_umap(self):
        """
        Generate UMAP visualization of the learned embeddings.
        
        Creates UMAP plots colored by predicted clusters, batch information,
        and ground truth labels (if available).
        """
        print('Start UMAP visualization')
        # Compute neighborhood graph and UMAP embedding
        sc.pp.neighbors(self.adata, use_rep='emb')
        sc.tl.umap(self.adata)
        
        # Generate UMAP plots with different colorings
        sc.pl.umap(self.adata, color='cluster', show=True, save=str(self.args.slide)+ 'cluster.pdf')
        sc.pl.umap(self.adata, color='batch', show=True, save=str(self.args.slide)+ '_batch.pdf')
        
        # Plot ground truth if labels are available
        if self.args.label==True:
            sc.pl.umap(self.adata, color='ground_truth', show=True, save=str(self.args.slide)+ '_label.pdf')
        