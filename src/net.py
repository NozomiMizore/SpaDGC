import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .utils import _nan2inf

class ZINB_decoder(torch.nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) decoder for gene expression reconstruction.
    
    This decoder reconstructs gene expression from learned embeddings using ZINB distribution,
    which is well-suited for modeling sparse count data in single-cell genomics.
    
    Args:
        nfeat (int): Number of features (genes)
        nhid1 (int): First hidden layer dimension  
        nhid2 (int): Input embedding dimension
    """
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINB_decoder, self).__init__()
        
        # Decoder network to transform embeddings
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        
        # ZINB distribution parameters
        self.pi = torch.nn.Linear(nhid1, nfeat)      # Zero-inflation probability
        self.disp = torch.nn.Linear(nhid1, nfeat)    # Dispersion parameter (theta)
        self.mean = torch.nn.Linear(nhid1, nfeat)    # Mean parameter (mu)
        
        # Activation functions with numerical stability
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        """
        Forward pass to predict ZINB parameters.
        
        Args:
            emb (torch.Tensor): Input embeddings
            
        Returns:
            tuple: (pi, disp, mean) - ZINB distribution parameters
        """
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))      # Zero-inflation probability in [0,1]
        disp = self.DispAct(self.disp(x))   # Dispersion parameter > 0
        mean = self.MeanAct(self.mean(x))   # Mean parameter > 0
        return pi, disp, mean
    
class Discriminator(nn.Module):
    """
    Discriminator network for neighborhood-level contrastive learning.

    This discriminator distinguishes between positive and negative pairs
    in the neighborhood-level contrastive learning framework, helping to learn 
    meaningful representations by maximizing agreement between positive pairs.
    
    Args:
        n_h (int): Hidden dimension size
    """
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        # Bilinear layer for computing similarity scores
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        # Initialize weights
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """Xavier uniform initialization for bilinear layers."""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """
        Forward pass to compute similarity scores.
        
        Args:
            c (torch.Tensor): Context vector (summary representation)
            h_pl (torch.Tensor): Positive samples
            h_mi (torch.Tensor): Negative samples  
            s_bias1 (torch.Tensor, optional): Bias for positive scores
            s_bias2 (torch.Tensor, optional): Bias for negative scores
            
        Returns:
            tuple: (positive_scores, negative_scores)
        """
        # Expand context to match sample dimensions
        c_x = c.expand_as(h_pl)

        # Compute similarity scores
        sc_1 = self.f_k(h_pl, c_x)  # Positive pair scores
        sc_2 = self.f_k(h_mi, c_x)  # Negative pair scores

        # Add bias terms if provided
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    """
    Average readout function for neighborhood representation.
    
    This module computes a neighborhood summary representation of the graph
    by taking the weighted average of node embeddings, where weights
    are determined by the adjacency matrix.
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        """
        Compute neighborhood representation via weighted averaging.
        
        Args:
            emb (torch.Tensor): Node embeddings [num_nodes, embedding_dim]
            mask (torch.Tensor): Adjacency matrix for weighting [num_nodes, num_nodes]
            
        Returns:
            torch.Tensor: L2-normalized neighborhood representation
        """
        # Weighted sum of embeddings using adjacency matrix
        vsum = torch.mm(mask, emb)
        
        # Calculate row sums for normalization
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        
        # Compute weighted average
        neighborhood_emb = vsum / row_sum

        # Return L2-normalized neighborhood embedding
        return F.normalize(neighborhood_emb, p=2, dim=1)

class Model_GCN(torch.nn.Module):
    """
    Graph Convolutional Network model for spatial transcriptomics analysis.
    
    This model combines a GCN encoder for learning node embeddings with:
    - ZINB decoder for gene expression reconstruction
    - Discriminator for contrastive learning
    - Readout function for neighborhood graph representation
    
    The model learns spatially-aware representations by leveraging both
    gene expression and spatial neighborhood information.
    
    Args:
        in_dim (int): Input feature dimension (number of genes)
        num_hidden (int): Hidden layer dimension
        out_dim (int): Output embedding dimension
    """
    def __init__(self, in_dim, num_hidden, out_dim):
        super(Model_GCN, self).__init__()

        # GCN encoder layers
        self.conv1 = GCNConv(in_dim, num_hidden)    # First GCN layer
        self.conv2 = GCNConv(num_hidden, out_dim)   # Second GCN layer
        
        # Decoder and auxiliary components
        self.ZINB_decoder = ZINB_decoder(in_dim, num_hidden, out_dim)  # Gene expression reconstruction
        self.disc = Discriminator(out_dim)          # Contrastive learning discriminator
        self.read = AvgReadout()                    # neighborhood representation

    def forward(self, features, features_shuffled, edge_index, adj, edge_weight=None):
        """
        Forward pass through the GCN model.
        
        Args:
            features (torch.Tensor): Original node features [num_nodes, in_dim]
            features_shuffled (torch.Tensor): Shuffled features for negative sampling
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            adj (torch.Tensor): Adjacency matrix [num_nodes, num_nodes]
            edge_weight (torch.Tensor, optional): Edge weights [num_edges]
            
        Returns:
            tuple: (embeddings, pi, disp, mean, contrastive_scores_1, contrastive_scores_2)
        """
        # Forward pass for original features
        h1 = F.relu(self.conv1(features, edge_index, edge_weight))          # First layer with ReLU
        h2 = self.conv2(h1, edge_index, edge_weight)                        # Second layer (embeddings)
        origin_emb = h2

        # Decode embeddings to reconstruct gene expression
        pi, disp, mean = self.ZINB_decoder(origin_emb)

        # Forward pass for shuffled features (negative samples)
        h1_shuffled = F.relu(self.conv1(features_shuffled, edge_index, edge_weight))
        h2_shuffled = self.conv2(h1_shuffled, edge_index, edge_weight)
        shuffled_emb = h2_shuffled

        # Generate neighborhood representations for contrastive learning
        summary = F.sigmoid(self.read(origin_emb, adj))           # Neighborhood representation of original graph
        summary_shuffled = F.sigmoid(self.read(shuffled_emb, adj)) # Neighborhood representation of shuffled graph

        # Compute contrastive learning scores
        ret = self.disc(summary, h2, h2_shuffled)           # Original vs shuffled discrimination
        ret_a = self.disc(summary_shuffled, h2_shuffled, h2) # Shuffled vs original discrimination

        return origin_emb, pi, disp, mean, ret, ret_a
    

class NB(object):
    """
    Negative Binomial (NB) distribution for modeling count data.
    
    This class implements the negative binomial loss function, which is
    commonly used for modeling gene expression count data in genomics.
    The NB distribution can handle overdispersion in count data better
    than Poisson distribution.
    
    Args:
        theta (torch.Tensor): Dispersion parameter (higher values = less overdispersion)
        scale_factor (float): Scaling factor for predictions
    """
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10              # Small constant for numerical stability
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        """
        Compute negative binomial loss.
        
        Args:
            y_true (torch.Tensor): True count values
            y_pred (torch.Tensor): Predicted mean values
            mean (bool): Whether to return mean loss or sum
            
        Returns:
            torch.Tensor: Negative binomial loss
        """
        # Scale predictions
        y_pred = y_pred * self.scale_factor
        
        # Clamp theta to prevent numerical issues
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        
        # Compute log-gamma terms for NB probability mass function
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        
        # Compute remaining terms of NB log-likelihood
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        
        # Combine terms and handle numerical issues
        final = t1 + t2
        final = _nan2inf(final)
        
        if mean:
            final = torch.mean(final)
        return final

class ZINB(NB):
    """
    Zero-Inflated Negative Binomial (ZINB) distribution for sparse count data.
    
    Extends the NB distribution to handle excess zeros commonly found in
    single-cell RNA sequencing data. The ZINB model assumes that zeros
    can arise from two sources: sampling zeros (from NB) and structural
    zeros (from zero-inflation process).
    
    Args:
        pi (torch.Tensor): Zero-inflation probability [0,1]
        ridge_lambda (float): Ridge regularization strength for pi
        **kwargs: Additional arguments passed to parent NB class
    """
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi                    # Zero-inflation probability
        self.ridge_lambda = ridge_lambda # Regularization strength

    def loss(self, y_true, y_pred, mean=True):
        """
        Compute zero-inflated negative binomial loss.
        
        This function handles two cases:
        1. Zero observations: Could be structural zeros (with prob pi) or sampling zeros
        2. Non-zero observations: Must come from NB distribution (not zero-inflated)
        
        Args:
            y_true (torch.Tensor): True count values
            y_pred (torch.Tensor): Predicted mean values  
            mean (bool): Whether to return mean loss or sum
            
        Returns:
            torch.Tensor: ZINB loss with optional ridge regularization
        """
        scale_factor = self.scale_factor
        eps = self.eps
        
        # Clamp theta for numerical stability
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        
        # Case 1: Non-zero observations - use NB likelihood adjusted for zero-inflation
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        
        # Case 2: Zero observations - mixture of structural and sampling zeros
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)  # P(Y=0|NB)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)  # P(Y=0|ZINB)
        
        # Select appropriate case based on whether observation is zero
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        
        # Add ridge regularization to prevent pi from becoming too large
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        
        if mean:
            result = torch.mean(result)
        
        # Handle numerical issues
        result = _nan2inf(result)
        return result