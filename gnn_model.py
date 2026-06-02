# gnn_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1. SMILES → Graph conversion
# ──────────────────────────────────────────────

ATOM_CHOICES = {
    'atomic_num': [6, 7, 8, 9, 5, 14, 15, 16, 17, 26, 28, 30, 34, 35, 44, 53, 78],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, 0, 1],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ],
    'is_aromatic': [False, True],
}

ATOM_FEAT_DIM = sum(len(v) for v in ATOM_CHOICES.values()) + 5

BOND_CHOICES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
}

BOND_FEAT_DIM = sum(len(v) for v in BOND_CHOICES.values())


def _one_hot(val, choices):
    vec = [0] * len(choices)
    if val in choices:
        vec[choices.index(val)] = 1
    return vec


def _atom_features(atom):
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_CHOICES['atomic_num'])
        + _one_hot(atom.GetDegree(), ATOM_CHOICES['degree'])
        + _one_hot(atom.GetFormalCharge(), ATOM_CHOICES['formal_charge'])
        + _one_hot(atom.GetHybridization(), ATOM_CHOICES['hybridization'])
        + _one_hot(atom.GetIsAromatic(), ATOM_CHOICES['is_aromatic'])
        + [atom.GetTotalNumHs()]
        + [atom.IsInRing() * 1.0]
        + [atom.GetNumRadicalElectrons()]
        + [atom.IsInRingSize(5) * 1.0]
        + [atom.IsInRingSize(6) * 1.0]
    )


def _bond_features(bond):
    return (
        _one_hot(bond.GetBondType(), BOND_CHOICES['bond_type'])
        + _one_hot(bond.GetIsConjugated(), BOND_CHOICES['is_conjugated'])
        + _one_hot(bond.IsInRing(), BOND_CHOICES['is_in_ring'])
    )


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()],
                      dtype=torch.float)

    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edges += [[i, j], [j, i]]
        edge_feats += [bf, bf]  # same features both directions

    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.zeros((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(edge_feats, dtype=torch.float)
                 if edge_feats else torch.zeros((0, BOND_FEAT_DIM)))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def precompute_graphs(smiles_series):
    """Returns a dict {smiles_string: Data}. Call once, reuse everywhere."""
    cache = {}
    for smi in smiles_series.unique():
        g = smiles_to_graph(smi)
        if g is not None:
            cache[smi] = g
    return cache


# ──────────────────────────────────────────────
# 2. Dataset — lightweight in-memory, no PyG Dataset overhead
# ──────────────────────────────────────────────

class DyeDeviceDataset:
    def __init__(self, graphs_list, device_array, targets):
        device_t = torch.tensor(device_array, dtype=torch.float)
        targets_t = torch.tensor(targets, dtype=torch.float)
        self.data_list = []
        for i, g in enumerate(graphs_list):
            d = g.clone()
            d.device_feat = device_t[i].unsqueeze(0)   # (1, n_features)
            d.y = targets_t[i].unsqueeze(0)             # (1,)
            self.data_list.append(d)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ──────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────

class DyeDeviceGNN(nn.Module):
    def __init__(
        self,
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        device_feat_dim=10,
        gnn_hidden=128,
        gnn_out=64,
        n_gnn_layers=3,
        device_hidden=32,
        pred_hidden=64,
        dropout=0.2,
    ):
        super().__init__()

        # GNN branch with edge features
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        self.gnn_layers.append(
            GATv2Conv(atom_feat_dim, gnn_hidden, edge_dim=bond_feat_dim)
        )
        self.gnn_norms.append(nn.LayerNorm(gnn_hidden))

        for _ in range(n_gnn_layers - 1):
            self.gnn_layers.append(
                GATv2Conv(gnn_hidden, gnn_hidden, edge_dim=bond_feat_dim)
            )
            self.gnn_norms.append(nn.LayerNorm(gnn_hidden))

        self.gnn_project = nn.Linear(gnn_hidden * 2, gnn_out)  # *2 for mean+max

        # Device branch
        self.device_mlp = nn.Sequential(
            nn.Linear(device_feat_dim, device_hidden),
            nn.LayerNorm(device_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(device_hidden, device_hidden),
        )

        # Prediction head
        combined_dim = gnn_out + device_hidden
        self.pred_head = nn.Sequential(
            nn.Linear(combined_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, pred_hidden // 2),
            nn.ReLU(),
            nn.Linear(pred_hidden // 2, 1),
        )
        self.dropout = dropout

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        edge_attr = batch.edge_attr

        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        dye_mean = global_mean_pool(x, batch_idx)
        dye_max = global_max_pool(x, batch_idx)
        dye_emb = self.gnn_project(torch.cat([dye_mean, dye_max], dim=1))

        dev_emb = self.device_mlp(batch.device_feat)

        combined = torch.cat([dye_emb, dev_emb], dim=1)
        return self.pred_head(combined).squeeze(-1)


# ──────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────

def train_gnn(model, train_ds, val_ds, lr=1e-3, epochs=100,
              batch_size=64, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(2, len(val_ds)))

    # Save initial state as fallback
    best_val = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_ok = True
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y)

            if not torch.isfinite(loss):
                epoch_ok = False
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if not epoch_ok:
            model.load_state_dict(best_state)
            return model

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                vl = F.mse_loss(model(batch), batch.y).item()
                val_losses.append(vl)
            val_loss = np.mean(val_losses)

        if not np.isfinite(val_loss):
            model.load_state_dict(best_state)
            return model

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────
# 5. Sklearn-compatible wrapper
# ──────────────────────────────────────────────

class GNNPipeline:
    def __init__(self, smiles_series, graph_cache, hparams,
                 impute_strategy='median', random_state=0):
        self.smiles_series = smiles_series
        self.graph_cache = graph_cache
        self.hparams = hparams
        self.impute_strategy = impute_strategy
        self.random_state = random_state
        self.model = None
        self.imputer = None
        self.scaler = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _prepare_device(self, X, fit=False):
        if fit:
            if self.impute_strategy is None:
                from sklearn.impute import KNNImputer
                self.imputer = KNNImputer(n_neighbors=5).fit(X)
            else:
                self.imputer = SimpleImputer(
                    strategy=self.impute_strategy
                ).fit(X)
            self.scaler = StandardScaler().fit(
                self.imputer.transform(X)
            )
        result = self.scaler.transform(self.imputer.transform(X))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _build_dataset(self, idx, device_arr, targets=None):
        smiles = self.smiles_series.loc[idx].values
        graphs = [self.graph_cache[s] for s in smiles]
        y = targets if targets is not None else np.zeros(len(idx))
        return DyeDeviceDataset(graphs, device_arr, y)

    def fit(self, X_train, y_train, max_epochs=100, patience=10):
        torch.manual_seed(self.random_state)
        dev_arr = self._prepare_device(X_train, fit=True)

        # Normalize targets
        self.y_mean = float(y_train.mean())
        self.y_std = float(y_train.std())
        if self.y_std < 1e-8:
            self.y_std = 1.0
        y_normalized = ((y_train - self.y_mean) / self.y_std).values

        train_ds = self._build_dataset(X_train.index, dev_arr, y_normalized)

        # Separate training params from architecture params
        excluded_keys = ('lr', 'bond_feat_dim')
        model_hparams = {k: v for k, v in self.hparams.items()
                         if k not in excluded_keys}
        lr = self.hparams.get('lr', 1e-3)

        self.model = DyeDeviceGNN(
            device_feat_dim=dev_arr.shape[1],
            bond_feat_dim=BOND_FEAT_DIM,
            **model_hparams,
        )

        # 90/10 train/val split for early stopping
        n_val = max(1, int(0.1 * len(train_ds)))
        n_tr = len(train_ds) - n_val
        tr_ds, vl_ds = torch.utils.data.random_split(
            train_ds, [n_tr, n_val],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        train_gnn(self.model, tr_ds, vl_ds,
                  lr=lr, epochs=max_epochs, patience=patience)
        return self

    def predict(self, X):
        dev_arr = self._prepare_device(X, fit=False)
        ds = self._build_dataset(X.index, dev_arr)
        loader = DataLoader(ds, batch_size=256)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch).detach().cpu().numpy()
                preds.append(out)

        result = np.concatenate(preds)

        # Replace any NaN/Inf with mean prediction (safe fallback)
        bad_mask = ~np.isfinite(result)
        if bad_mask.any():
            result[bad_mask] = 0.0

        # Denormalize
        return result * self.y_std + self.y_mean