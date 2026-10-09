# ============================================================
# 03_gnn_model.py
# Shared GCN encoder + condition-specific ASD/TD heads
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne

from scipy.spatial.distance import cdist


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_spatial_graph(
    channels,
    k=4
):
    """
    Build a fixed k-nearest-neighbor graph using
    standard 10-20 electrode coordinates.

    Returns
    -------
    adjacency : torch.FloatTensor
        [N, N], normalized adjacency with self-loops

    coords : torch.FloatTensor
        [N, 3], standardized XYZ coordinates
    """

    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )

    positions = montage.get_positions()["ch_pos"]

    # case-insensitive lookup
    position_lookup = {
        key.upper(): value
        for key, value in positions.items()
    }

    coords = []

    missing = []

    for ch in channels:

        ch_upper = str(ch).upper()

        if ch_upper not in position_lookup:
            missing.append(ch)
            continue

        coords.append(
            position_lookup[ch_upper]
        )

    if missing:
        raise ValueError(
            f"Channels missing from standard_1020 montage: "
            f"{missing}"
        )

    coords = np.asarray(
        coords,
        dtype=np.float32
    )

    n_nodes = len(channels)

    # --------------------------------------------------------
    # k-NN graph
    # --------------------------------------------------------

    distances = cdist(
        coords,
        coords
    )

    adjacency = np.zeros(
        (n_nodes, n_nodes),
        dtype=np.float32
    )

    for i in range(n_nodes):

        # first one is self
        nearest = np.argsort(
            distances[i]
        )[1:k + 1]

        adjacency[i, nearest] = 1.0

    # make symmetric
    adjacency = np.maximum(
        adjacency,
        adjacency.T
    )

    # self-loops
    adjacency += np.eye(
        n_nodes,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # GCN normalization
    #
    # D^-1/2 A D^-1/2
    # --------------------------------------------------------

    degree = adjacency.sum(
        axis=1
    )

    degree_inv_sqrt = (
        1.0
        / np.sqrt(
            degree + 1e-8
        )
    )

    D_inv_sqrt = np.diag(
        degree_inv_sqrt
    )

    adjacency_norm = (
        D_inv_sqrt
        @ adjacency
        @ D_inv_sqrt
    )

    # --------------------------------------------------------
    # standardize XYZ coordinates
    # --------------------------------------------------------

    coord_mean = coords.mean(
        axis=0,
        keepdims=True
    )

    coord_std = coords.std(
        axis=0,
        keepdims=True
    )

    coord_std[
        coord_std < 1e-8
    ] = 1.0

    coords_z = (
        coords - coord_mean
    ) / coord_std

    return (
        torch.tensor(
            adjacency_norm,
            dtype=torch.float32
        ),
        torch.tensor(
            coords_z,
            dtype=torch.float32
        )
    )


# ============================================================
# GCN LAYER
# ============================================================

class GCNLayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )


    def forward(
        self,
        x,
        adjacency
    ):
        """
        x:
            [batch_trials, nodes, features]

        adjacency:
            [nodes, nodes]
        """

        # neighborhood aggregation
        x = torch.einsum(
            "ij,bjf->bif",
            adjacency,
            x
        )

        x = self.linear(
            x
        )

        return x


# ============================================================
# SHARED GNN
# ============================================================

class SharedConditionGCN(nn.Module):

    def __init__(
        self,
        adjacency,
        node_coords,
        n_conditions,
        psd_features=5,
        hidden_dim=32,
        embedding_dim=32,
        dropout=0.30
    ):

        super().__init__()

        self.register_buffer(
            "adjacency",
            adjacency
        )

        self.register_buffer(
            "node_coords",
            node_coords
        )

        self.n_conditions = (
            n_conditions
        )

        # PSD 4 + XYZ 3
        input_dim = (
            psd_features + 3
        )

        self.gcn1 = GCNLayer(
            input_dim,
            hidden_dim
        )

        self.gcn2 = GCNLayer(
            hidden_dim,
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # mean + max graph pooling
        self.trial_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            )
        )

        # one binary classifier per condition
        self.heads = nn.ModuleList([
            nn.Linear(
                embedding_dim,
                1
            )
            for _ in range(
                n_conditions
            )
        ])


    def encode_trials(
        self,
        x
    ):
        """
        x:
            [B, T, N, 4]

        Returns
        -------
        trial_embeddings:
            [B, T, embedding_dim]
        """

        B, T, N, F_psd = (
            x.shape
        )

        # ----------------------------------------
        # append spatial XYZ coordinates
        # ----------------------------------------

        coords = (
            self.node_coords
            .view(
                1,
                1,
                N,
                3
            )
            .expand(
                B,
                T,
                N,
                3
            )
        )

        x = torch.cat(
            [
                x,
                coords
            ],
            dim=-1
        )

        # flatten subject-condition bags
        # into independent trial graphs
        x = x.reshape(
            B * T,
            N,
            -1
        )

        # ----------------------------------------
        # GCN
        # ----------------------------------------

        x = self.gcn1(
            x,
            self.adjacency
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        x = self.gcn2(
            x,
            self.adjacency
        )

        x = F.relu(
            x
        )

        # ----------------------------------------
        # graph pooling across electrodes
        # ----------------------------------------

        mean_pool = x.mean(
            dim=1
        )

        max_pool = x.max(
            dim=1
        ).values

        x = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )

        trial_embedding = (
            self.trial_projection(
                x
            )
        )

        trial_embedding = (
            trial_embedding.reshape(
                B,
                T,
                -1
            )
        )

        return trial_embedding


    def forward(
        self,
        x,
        trial_mask,
        condition_idx
    ):
        """
        x:
            [B, T, N, 4]

        trial_mask:
            [B, T]

        condition_idx:
            [B]

        Returns
        -------
        logits:
            [B]

        bag_embedding:
            [B, embedding_dim]
        """

        trial_embeddings = (
            self.encode_trials(
                x
            )
        )

        # ----------------------------------------
        # subject-condition mean pooling
        # ----------------------------------------

        mask = (
            trial_mask
            .unsqueeze(-1)
            .float()
        )

        summed = (
            trial_embeddings
            * mask
        ).sum(
            dim=1
        )

        denominator = (
            mask.sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )

        bag_embedding = (
            summed
            / denominator
        )

        # ----------------------------------------
        # calculate all condition-head logits
        # ----------------------------------------

        all_logits = torch.cat(
            [
                head(
                    bag_embedding
                )
                for head in self.heads
            ],
            dim=1
        )

        # choose correct head for each sample
        logits = (
            all_logits
            .gather(
                1,
                condition_idx
                .view(-1, 1)
            )
            .squeeze(1)
        )

        return (
            logits,
            bag_embedding
        )