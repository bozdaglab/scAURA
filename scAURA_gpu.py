import os, json, warnings, math, contextlib
from itertools import product
from typing import Optional, List, Tuple
import time
from datetime import datetime

import numpy as np
import pandas as pd
import random

from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import GCL.augmentors as A
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import coo_matrix

warnings.filterwarnings("ignore", category=UserWarning)



SEED = 42
K_CLUSTERS = 11                 
N_JOBS = -1                    
K_TUNE_JOBS = 5                

K_TUNE = True                   
K_MIN = 10                      
K_MAX = 11                      
DEC_INIT_METHODS = ["kmeans", "spectral"]
KMEANS_N_INIT = 20
SPECTRAL_N_NEIGHBORS_GRID = [30]
SPECTRAL_ASSIGN_LABELS = "kmeans"

PARAM_GRID = {
    "USE_PCA_FOR_GRAPH": [True],        
    "PCA_N_PCS":          [40],         
    "AK_KMAX":            [40],         
    "AK_DELTA":           [2],        
    "GRAPH_METRIC":       ['euclidean'],

    "HIDDEN_DIM":         [64],         
    "PROJ_DIM":           [32],         
    "NUM_GCN_LAYERS":     [2],          
    "ACTIVATION":         ['tanh','relu'],     

    "PE":                 [0.1],        
    "PF":                 [0.1],        
    "TAU":                [0.6,0.7,0.8],        
    "TAU_PLUS":           [0.3,0.4],        
    "LR":                 [1e-3],       
    "EPOCHS":             [200],        

    "LAMBDA_ALIGN":       [1.0],        
    "LAMBDA_UNIFORM":     [1.0],        
    "UNIFORM_WARMUP":     [20],         
    "NORMALIZE_EMBEDS":   [True],      
    "DEC_BATCHSIZE":      [32,64,128],
}


REP_SEEDS = [101, 370, 124, 319, 2024, 8080, 4142, 141, 371, 421]

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


SELF_TRAIN = True
DEC_ALPHA = 1.0
DEC_LR = 1e-3
DEC_MAX_EPOCHS = 500
DEC_UPDATE_ITERS = 10
DEC_TOL = 0.005

DEC_SGD_MOMENTUM = 0.9
DEC_SGD_WEIGHT_DECAY = 0.0
DEC_SGD_NESTEROV = True


def get_devices_for_workers() -> List[int]:
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    return list(range(n))

def pick_device(gpu_id: Optional[int]):
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@contextlib.contextmanager
def maybe_autocast(device: torch.device):
    use_cuda = (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_cuda, dtype=torch.bfloat16):
        yield


torch.manual_seed(SEED); np.random.seed(SEED)


de_df_temp = pd.read_csv(Path(__file__).parent / "dataset"/"Klein"/ "klein.csv")
X_np = de_df_temp.iloc[:, 1:501].values.astype(np.float32)   
labels = de_df_temp['label']
le = preprocessing.LabelEncoder()
y_np = le.fit_transform(labels).astype(np.int64)
num_features = X_np.shape[1]

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super().__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

class Encoder(nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super().__init__()
        self.encoder = encoder
        self.augmentor = augmentor if augmentor is not None else (None, None)
        self.fc1 = nn.Linear(hidden_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, hidden_dim)
    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        if aug1 is not None and aug2 is not None:
            x1, ei1, ew1 = aug1(x, edge_index, edge_weight)
            x2, ei2, ew2 = aug2(x, edge_index, edge_weight)
        else:
            x1, ei1, ew1 = x, edge_index, edge_weight
            x2, ei2, ew2 = x, edge_index, edge_weight
        z  = self.encoder(x,  edge_index,  edge_weight)
        z1 = self.encoder(x1, ei1, ew1)
        z2 = self.encoder(x2, ei2, ew2)
        return z, z1, z2
    def project(self, z): return self.fc2(F.elu(self.fc1(z)))

def get_act(name:str):
    name = name.lower()
    lut = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}
    if name not in lut:
        raise ValueError(f"Unsupported ACTIVATION '{name}'. Use one of: {list(lut.keys())}")
    return lut[name]

def _choose_adaptive_k(d_row_sorted, Kmax, delta):
    denom = max(1e-8, (Kmax - 1 - delta))
    T = (np.sum(np.sqrt(d_row_sorted)) / denom) ** 2
    k = np.searchsorted(d_row_sorted, T, side='right')
    return max(1, int(k))

def _build_adaptive_knn(X, Kmax, delta, use_pca, n_pcs, metric, random_state=SEED):
    if use_pca:
        pca = PCA(n_components=min(n_pcs, X.shape[1]), random_state=random_state)
        X_ = pca.fit_transform(X)
    else:
        X_ = X
    nnbr = NearestNeighbors(n_neighbors=Kmax, metric=metric)
    nnbr.fit(X_)
    dists, nbrs = nnbr.kneighbors(X_)
    N = X.shape[0]
    ks = np.array([_choose_adaptive_k(np.sort(dists[i]), Kmax=Kmax, delta=delta) for i in range(N)], dtype=int)
    rows = np.repeat(np.arange(N), ks)
    cols = np.concatenate([nbrs[i, :ks[i]] for i in range(N)])
    return rows, cols, nbrs, ks

def _snn_jaccard(rows, cols, nbrs, ks, symmetric=True):
    N = nbrs.shape[0]
    neigh_sets = [set(nbrs[i, :ks[i]].tolist()) for i in range(N)]
    weights = np.empty(rows.shape[0], dtype=float)
    for idx, (i, j) in enumerate(zip(rows, cols)):
        Ni, Nj = neigh_sets[i], neigh_sets[j]
        inter = len(Ni & Nj); uni = len(Ni | Nj) if (Ni or Nj) else 1
        weights[idx] = inter / max(1, uni)
    W = coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()
    if symmetric: W = W.maximum(W.T)
    return W

def build_full_graph(cfg, device: torch.device, random_state: int):
    rows, cols, nbrs, ks = _build_adaptive_knn(
        X_np,
        Kmax=cfg["AK_KMAX"],
        delta=cfg["AK_DELTA"],
        use_pca=cfg["USE_PCA_FOR_GRAPH"],
        n_pcs=cfg["PCA_N_PCS"],
        metric=cfg["GRAPH_METRIC"],
        random_state=random_state,   
    )
    W = _snn_jaccard(rows, cols, nbrs, ks, symmetric=True).tocoo()
    edge_index = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(W.data, dtype=torch.float32, device=device)
    data = Data(
        x=torch.tensor(X_np, dtype=torch.float32, device=device),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y_np, dtype=torch.long, device=device)
    )
    return data



def debiased_infonce_full_tiled(
    h1n: torch.Tensor,   
    h2n: torch.Tensor,   
    tau: float,
    tau_plus: float,
    device: torch.device,
    tile_rows: int = 4096,
    tile_cols: int = 4096,
) -> torch.Tensor:
    N, D = h1n.shape
    M = 2 * N
    emb = torch.cat([h1n, h2n], dim=0)  

    pos_sim = (h1n * h2n).sum(dim=1)                 
    pos_exp = torch.exp(pos_sim / tau)               
    pos_exp_all = torch.cat([pos_exp, pos_exp], 0)  

    rows_idx = torch.arange(M, device=device)
    pos_idx_all = rows_idx + N
    pos_idx_all[pos_idx_all >= M] -= M

    neg_sum = torch.zeros(M, device=device, dtype=torch.float32)
    col_idx_full = torch.arange(M, device=device)

    with maybe_autocast(device):
        for i0 in range(0, M, tile_rows):
            i1 = min(i0 + tile_rows, M)
            X = emb[i0:i1]                            
            pos_idx_tile = pos_idx_all[i0:i1]         
            partial = torch.zeros((i1-i0,), device=device, dtype=torch.float32)

            for j0 in range(0, M, tile_cols):
                j1 = min(j0 + tile_cols, M)
                Y = emb[j0:j1]                        

                logits = (X @ Y.t()) / tau            
                exp_block = torch.exp(logits)

                
                if i0 <= j1 and j0 <= i1:
                    r = torch.arange(i1 - i0, device=device).unsqueeze(1)
                    c = torch.arange(j1 - j0, device=device).unsqueeze(0)
                    self_mask = (i0 + r) == (j0 + c)
                    exp_block = exp_block.masked_fill(self_mask, 0)

                
                cols = col_idx_full[j0:j1]            
                pos_mask = (pos_idx_tile.unsqueeze(1) == cols.unsqueeze(0))
                exp_block = exp_block.masked_fill(pos_mask, 0)

                partial = partial + exp_block.sum(dim=1).float()

            neg_sum[i0:i1] += partial

    Np = M - 2
    Ng = (-tau_plus * Np * pos_exp_all + neg_sum) / (1.0 - tau_plus)
    Ng = torch.clamp(Ng, min=Np * math.e ** (-1.0 / tau))
    loss = (-torch.log(pos_exp_all / (pos_exp_all + Ng))).mean()
    return loss

def uniformity_allpairs_tiled(
    emb: torch.Tensor,            
    alpha: float,                 
    device: torch.device,
    tile_rows: int = 4096,
    tile_cols: int = 4096,
    use_upper_triangle: bool = True,
) -> torch.Tensor:

    M, D = emb.shape
    emb_f32 = emb if emb.dtype == torch.float32 else emb.float()
    norms = (emb_f32**2).sum(dim=1, keepdim=True)  

    total_sum = emb_f32.new_zeros(())
    total_cnt = 0
    col_start = (lambda i: i) if use_upper_triangle else (lambda i: 0)

    with maybe_autocast(device):
        for i0 in range(0, M, tile_rows):
            i1 = min(i0 + tile_rows, M)
            X = emb[i0:i1]          
            X2 = norms[i0:i1]       

            j_begin = col_start(i0)
            for j0 in range(j_begin, M, tile_cols):
                j1 = min(j0 + tile_cols, M)
                Y = emb[j0:j1]      
                Y2 = norms[j0:j1].transpose(0,1) 

                dist2 = X2 + Y2 - 2.0 * (X @ Y.t())  

                if i0 == j0:
                    d = torch.arange(0, i1 - i0, device=device)
                    dist2[d, d] = float("inf")       
                    cnt_block = dist2.numel() - (i1 - i0)
                else:
                    cnt_block = dist2.numel()

                s_block = torch.exp(-alpha * dist2).sum()
                total_sum = total_sum + s_block
                total_cnt += cnt_block

    mean_val = total_sum / max(1, total_cnt)
    return torch.log(mean_val)


def t_student_q(z: torch.Tensor, mu: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    z2 = (z**2).sum(dim=1, keepdim=True)
    m2 = (mu**2).sum(dim=1, keepdim=True).t()
    dist2 = z2 + m2 - 2.0 * (z @ mu.t())
    num = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
    q = num / (num.sum(dim=1, keepdim=True) + 1e-12)
    return q

@torch.no_grad()
def target_distribution(q: torch.Tensor) -> torch.Tensor:
    f = q.sum(dim=0, keepdim=True)
    w = (q ** 2) / (f + 1e-12)
    p = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return p

def kl_divergence_pq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum() / p.size(0)

def dec_refine(
    encoder: nn.Module,
    data: Data,
    init_centers: np.ndarray,
    alpha: float = DEC_ALPHA,
    lr: float = DEC_LR,
    max_epochs: int = DEC_MAX_EPOCHS,
    update_interval: int = DEC_UPDATE_ITERS,
    tol: float = DEC_TOL,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
):
    encoder.train()
    N = data.x.size(0)

    mu = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32, device=device))
    opt = torch.optim.SGD(
        list(encoder.parameters()) + [mu],
        lr=lr,
        momentum=DEC_SGD_MOMENTUM,
        weight_decay=DEC_SGD_WEIGHT_DECAY,
        nesterov=DEC_SGD_NESTEROV,
    )

    prev_y = None
    hist = {"kl": [], "changed_frac": []}

    def forward_all():
        encoder.eval()
        with torch.no_grad():
            z, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
        encoder.train()
        return z

    z = forward_all()
    q = t_student_q(z, mu, alpha=alpha)
    p = target_distribution(q)
    prev_y = torch.argmax(q, dim=1)

    for epoch in range(1, max_epochs + 1):
        if batch_size is None:
            z, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
            q = t_student_q(z, mu, alpha=alpha)
            loss = kl_divergence_pq(p, q)
            opt.zero_grad(); loss.backward(); opt.step()
            hist["kl"].append(float(loss.item()))
        else:
            perm = torch.randperm(N, device=device)
            epoch_loss = 0.0
            
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                z_all, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
                zb = z_all[idx]
                qb = t_student_q(zb, mu, alpha=alpha)
                pb = p[idx]
                lb = kl_divergence_pq(pb, qb)
                opt.zero_grad(); lb.backward(); opt.step()
                epoch_loss += float(lb.item())
            hist["kl"].append(epoch_loss)

        if epoch % update_interval == 0 or epoch == max_epochs:
            z = forward_all()
            q = t_student_q(z, mu, alpha=alpha)
            p = target_distribution(q)
            y = torch.argmax(q, dim=1)
            changed = (y != prev_y).sum().item()
            changed_frac = changed / float(N)
            hist["changed_frac"].append(changed_frac)
            prev_y = y
            if changed_frac < tol:
                break

    z = forward_all()
    q = t_student_q(z, mu, alpha=alpha)
    y = torch.argmax(q, dim=1)

    return y.detach().cpu().numpy(), mu.detach().cpu().numpy(), {
        "q": q.detach().cpu().numpy(),
        "changed_frac": hist["changed_frac"],
        "kl": hist["kl"],
        "epochs_ran": epoch
    }


def train_one(cfg, run_dir=None, save_metrics=True, gpu_id: Optional[int]=None, base_seed: Optional[int]=None):
    
    seed = base_seed if base_seed is not None else SEED

    device = pick_device(gpu_id)
    seed_all(seed)

    data = build_full_graph(cfg, device, random_state=seed)

    gconv = GConv(
        input_dim=num_features,
        hidden_dim=cfg["HIDDEN_DIM"],
        activation=get_act(cfg["ACTIVATION"]),
        num_layers=cfg["NUM_GCN_LAYERS"]
    ).to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=cfg["PE"]), A.FeatureMasking(pf=cfg["PF"])])
    aug2 = A.Compose([A.EdgeRemoving(pe=cfg["PE"]), A.FeatureMasking(pf=cfg["PF"])])

    encoder = Encoder(gconv, (aug1, aug2), hidden_dim=cfg["HIDDEN_DIM"], proj_dim=cfg["PROJ_DIM"]).to(device)
    opt = Adam(encoder.parameters(), lr=cfg["LR"])

    loss_hist, align_hist, uni_hist, tot_hist, wu_hist = [], [], [], [], []

    for epoch in range(cfg["EPOCHS"]):
        encoder.train()
        with maybe_autocast(device):
            z, z1, z2 = encoder(data.x, data.edge_index, data.edge_attr)
            h1, h2 = encoder.project(z1), encoder.project(z2)
            if cfg["NORMALIZE_EMBEDS"]:
                h1n, h2n = F.normalize(h1, dim=1), F.normalize(h2, dim=1)
            else:
                h1n, h2n = h1, h2

        align = (h1n - h2n).pow(2).sum(dim=1).mean()

        
        contrastive = debiased_infonce_full_tiled(
            h1n, h2n,
            tau=cfg["TAU"],
            tau_plus=cfg["TAU_PLUS"],
            device=device,
            tile_rows=4096, tile_cols=4096
        )

        
        emb_full = torch.cat([h1n, h2n], dim=0)
        uniformity = uniformity_allpairs_tiled(
            emb_full, alpha=2.0, device=device, tile_rows=4096, tile_cols=4096, use_upper_triangle=True
        )

        
        wu = min(1.0, (epoch + 1) / float(max(1, cfg["UNIFORM_WARMUP"]))) if cfg["UNIFORM_WARMUP"] > 0 else 1.0
        total_loss_epoch = contrastive + cfg["LAMBDA_ALIGN"] * align + (cfg["LAMBDA_UNIFORM"] * wu) * uniformity

        opt.zero_grad()
        total_loss_epoch.backward()
        opt.step()

        
        loss_hist.append(float(contrastive.item()))
        align_hist.append(float(align.item()))
        uni_hist.append(float(uniformity.item()))
        tot_hist.append(float(total_loss_epoch.item()))
        wu_hist.append(float(wu))

    encoder.eval()
    with torch.no_grad():
        z_pre, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
    embeddings_pre = z_pre.detach().cpu().numpy()

    y_true = y_np

    def compute_centers_from_labels(embeddings: np.ndarray, labels_pred: np.ndarray, k: int) -> np.ndarray:
        centers = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
        global_center = embeddings.mean(axis=0).astype(np.float32)
        for cluster_id in range(k):
            mask = labels_pred == cluster_id
            if np.any(mask):
                centers[cluster_id] = embeddings[mask].mean(axis=0)
            else:
                centers[cluster_id] = global_center
        return centers.astype(np.float32)

    def eval_init(method, k, n_neighbors=None):
        if method == "kmeans":
            model = KMeans(
                n_clusters=int(k),
                n_init=KMEANS_N_INIT,
                random_state=seed,
            )
            labels_pred = model.fit_predict(embeddings_pre)
            centers = model.cluster_centers_.astype(np.float32)
            used_n_neighbors = None
        elif method == "spectral":
            used_n_neighbors = min(int(n_neighbors), embeddings_pre.shape[0] - 1)
            model = SpectralClustering(
                n_clusters=int(k),
                affinity="nearest_neighbors",
                n_neighbors=used_n_neighbors,
                assign_labels=SPECTRAL_ASSIGN_LABELS,
                random_state=seed,
            )
            labels_pred = model.fit_predict(embeddings_pre)
            centers = compute_centers_from_labels(embeddings_pre, labels_pred, int(k))
        else:
            raise ValueError(f"Unsupported DEC initialization method: {method}")

        try:
            sil = (
                silhouette_score(embeddings_pre, labels_pred, metric="euclidean")
                if len(np.unique(labels_pred)) >= 2
                else float("-inf")
            )
        except Exception:
            sil = float("-inf")

        return {
            "method": method,
            "K": int(k),
            "n_neighbors": used_n_neighbors,
            "silhouette": float(sil),
            "labels": labels_pred,
            "centers": centers,
        }

    if K_TUNE:
        combos = []
        for method in DEC_INIT_METHODS:
            for k in range(K_MIN, K_MAX + 1):
                if method == "kmeans":
                    combos.append((method, k, None))
                elif method == "spectral":
                    for n_neighbors in SPECTRAL_N_NEIGHBORS_GRID:
                        combos.append((method, k, n_neighbors))

        results = Parallel(n_jobs=K_TUNE_JOBS, verbose=0)(
            delayed(eval_init)(method, k, n_neighbors)
            for (method, k, n_neighbors) in combos
        )

        df_k = pd.DataFrame([
            {
                "method": r["method"],
                "K": r["K"],
                "n_neighbors": r["n_neighbors"],
                "silhouette": r["silhouette"],
            }
            for r in results
        ]).sort_values(["silhouette", "method", "K"], ascending=[False, True, True], ignore_index=True)

        best_result = max(results, key=lambda r: (r["silhouette"], -r["K"]))
        best_init_method = best_result["method"]
        best_k = int(best_result["K"])
        best_n_neighbors = best_result["n_neighbors"]
        best_silhouette_init = float(best_result["silhouette"])
        y_pred = best_result["labels"]
        init_centers = best_result["centers"]
    else:
        fixed_results = []
        for method in DEC_INIT_METHODS:
            if method == "kmeans":
                fixed_results.append(eval_init(method, K_CLUSTERS, None))
            elif method == "spectral":
                for n_neighbors in SPECTRAL_N_NEIGHBORS_GRID:
                    fixed_results.append(eval_init(method, K_CLUSTERS, n_neighbors))

        best_result = max(fixed_results, key=lambda r: r["silhouette"])
        best_init_method = best_result["method"]
        best_k = int(best_result["K"])
        best_n_neighbors = best_result["n_neighbors"]
        best_silhouette_init = float(best_result["silhouette"])
        y_pred = best_result["labels"]
        init_centers = best_result["centers"]
        df_k = pd.DataFrame([
            {
                "method": r["method"],
                "K": r["K"],
                "n_neighbors": r["n_neighbors"],
                "silhouette": r["silhouette"],
            }
            for r in fixed_results
        ])
        
    y_pred_initial = y_pred.copy()
    q_soft = None
    dec_hist = None

    if SELF_TRAIN:
        y_refined, centers_refined, dec_details = dec_refine(
            encoder=encoder, data=data, init_centers=init_centers,
            alpha=DEC_ALPHA, lr=DEC_LR, max_epochs=DEC_MAX_EPOCHS,
            update_interval=DEC_UPDATE_ITERS, tol=DEC_TOL,
            batch_size=cfg["DEC_BATCHSIZE"], device=device,
        )
        y_pred = y_refined
        q_soft = dec_details.get("q", None)
        dec_hist = dec_details

    encoder.eval()
    with torch.no_grad():
        z_final, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
    embeddings_final = z_final.detach().cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    try:
        asw = silhouette_score(embeddings_final, y_pred, metric='euclidean') if len(np.unique(y_pred)) >= 2 else float('nan')
    except Exception:
        asw = float('nan')


    if save_metrics and run_dir:
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame({
            "epoch": np.arange(1, len(loss_hist)+1),
            "contrastive_loss": loss_hist,
            "alignment": align_hist,
            "uniformity": uni_hist,
            "total_loss": tot_hist,
            "uniformity_weight": wu_hist
        }).to_csv(os.path.join(run_dir, "training_metrics.csv"), index=False)

        if K_TUNE:
            df_k.to_csv(os.path.join(run_dir, "dec_init_tuning.csv"), index=False)

        np.save(os.path.join(run_dir, "embeddings_pre_dec.npy"), embeddings_pre)
        np.save(os.path.join(run_dir, "embeddings_post_dec.npy"), embeddings_final)

        pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv(os.path.join(run_dir, "spectral_init_dec_labels_final.csv"), index=False)

        summary = {
            "ARI": float(ari), "NMI": float(nmi), "ASW": float(asw),
            "BEST_K": int(best_k),
            "BEST_DEC_INIT_METHOD": str(best_init_method),
            "BEST_SPECTRAL_N_NEIGHBORS": (int(best_n_neighbors) if best_n_neighbors is not None else None),
            "BEST_INIT_SILHOUETTE": float(best_silhouette_init),
            "SELF_TRAIN": bool(SELF_TRAIN),
            "DEC_ALPHA": float(DEC_ALPHA),
            "DEC_TOL": float(DEC_TOL),
            "GPU_ID": gpu_id if gpu_id is not None else -1,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"cfg": cfg, "summary": summary}, f, indent=2)

        if SELF_TRAIN:
            if q_soft is not None:
                np.save(os.path.join(run_dir, "dec_Q_soft.npy"), q_soft)
            if dec_hist is not None:
                with open(os.path.join(run_dir, "dec_history.json"), "w") as f:
                    json.dump({
                        "changed_frac": dec_hist.get("changed_frac", []),
                        "kl": dec_hist.get("kl", []),
                        "epochs_ran": dec_hist.get("epochs_ran", 0)
                    }, f, indent=2)
            pd.DataFrame({
                "true": y_true,
                "spectral_init": y_pred_initial,
                "refined": y_pred
            }).to_csv(os.path.join(run_dir, "dec_labels_refined.csv"), index=False)

    return {
        "cfg": cfg,
        "best_k": int(best_k),
        "best_dec_init_method": str(best_init_method),
        "best_spectral_n_neighbors": (int(best_n_neighbors) if best_n_neighbors is not None else None),
        "best_init_silhouette": float(best_silhouette_init),
        "ari": float(ari),
        "nmi": float(nmi),
        "asw": float(asw)
    }


def expand_grid(param_grid: dict):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))

def main():
    NUM_REPEATS = 10
    assert NUM_REPEATS == len(REP_SEEDS)

    dataset_name = os.path.splitext(os.path.basename(os.path.expanduser(DATA_CSV)))[0]
    out_dir = f"res_grid_seed_{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    gpus = get_devices_for_workers()
    if not gpus:
        print("No CUDA device found, running on CPU. (It will be slower.)")
    num_gpus = max(1, len(gpus))

    n_jobs = num_gpus if N_JOBS in (-1, 0) else max(1, N_JOBS)
    print(f"Using {n_jobs} parallel workers across {num_gpus} GPUs: {gpus}")

    all_ari, all_nmi, all_asw = [], [], []

    grid = list(expand_grid(PARAM_GRID))
    print(f"Total experiments per repeat: {len(grid)}")

    for rep in range(NUM_REPEATS):
        rep_seed = REP_SEEDS[rep]
        print(f"\n=== RUN {rep+1}/{NUM_REPEATS}  (seed={rep_seed}) ===")

        rep_dir = os.path.join(out_dir, f"rep_{rep:02d}")
        os.makedirs(rep_dir, exist_ok=True)

        assign = [
            (i, grid[i], (gpus[i % num_gpus] if gpus else None), rep_seed)
            for i in range(len(grid))
        ]

        def run_indexed(payload: Tuple[int, dict, Optional[int], int]):
            idx, cfg, gpu_id, seed = payload
            run_dir = os.path.join(rep_dir, f"exp_{idx:04d}")
            res = train_one(cfg, run_dir=run_dir, save_metrics=True, gpu_id=gpu_id, base_seed=seed)
            res["exp_id"] = idx
            return res

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_indexed)(assign[i]) for i in range(len(assign))
        )

        df = pd.DataFrame([
            {
                **r["cfg"],
                "exp_id": r["exp_id"],
                "ASW": r["asw"],
                "ARI": r["ari"],
                "NMI": r["nmi"],
                "BEST_K": r["best_k"],
                "BEST_DEC_INIT_METHOD": r["best_dec_init_method"],
                "BEST_SPECTRAL_N_NEIGHBORS": r["best_spectral_n_neighbors"],
                "BEST_INIT_SILHOUETTE": r["best_init_silhouette"],
            }
            for r in results
        ]).sort_values(["ARI", "NMI", "ASW"], ascending=False, ignore_index=True)

        df.to_csv(os.path.join(rep_dir, "grid_results.csv"), index=False)

        best = df.iloc[0]
        all_ari.append(best["ARI"])
        all_nmi.append(best["NMI"])
        all_asw.append(best["ASW"])

        print(
            f"Rep {rep+1}: ASW={best['ASW']:.4f}, "
            f"ARI={best['ARI']:.4f}, "
            f"NMI={best['NMI']:.4f}"
        )

    print("\n=== SUMMARY OF 10 RUNS ===")
    print("ARI mean/std:", np.mean(all_ari), np.std(all_ari))
    print("NMI mean/std:", np.mean(all_nmi), np.std(all_nmi))
    print("ASW mean/std:", np.mean(all_asw), np.std(all_asw))

    summary_df = pd.DataFrame({
        "ARI": all_ari,
        "NMI": all_nmi,
        "ASW": all_asw
    })
    summary_df.loc["mean"] = summary_df.mean(numeric_only=True)
    summary_df.loc["std"] = summary_df.std(numeric_only=True)
    summary_df.to_csv(os.path.join(out_dir, "summary_10runs.csv"))

if __name__ == "__main__":
    main()
