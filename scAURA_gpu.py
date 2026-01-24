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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import GCL.augmentors as A
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import coo_matrix

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
import matplotlib.pyplot as plt
from pathlib import Path


SEED = 42
K_CLUSTERS = 4                 
N_JOBS = -1                     
K_TUNE_JOBS = -1                


K_TUNE = False                   
K_MIN = 3                       
K_MAX = 7                      

KMEANS_N_INIT_GRID = [10, 50, 100]
KMEANS_MAX_ITER_GRID = [300, 500]


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
    "UNIFORM_WARMUP":     [200],         
    "NORMALIZE_EMBEDS":   [True],       
    "SSC_BATCHSIZE":      [32,64,128],
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
SSC_ALPHA = 1.0
SSC_LR = 1e-3
SSC_MAX_EPOCHS = 500
SSC_UPDATE_ITERS = 10
SSC_TOL = 0.005


SSC_SGD_MOMENTUM = 0.9
SSC_SGD_WEIGHT_DECAY = 0.0
SSC_SGD_NESTEROV = True


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

de_df_temp = de_df_temp = pd.read_csv(Path(__file__).parent / "dataset"/"Klein"/ "klein.csv")
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

def ssc_refine(
    encoder: nn.Module,
    data: Data,
    init_centers: np.ndarray,
    alpha: float = SSC_ALPHA,
    lr: float = SSC_LR,
    max_epochs: int = SSC_MAX_EPOCHS,
    update_interval: int = SSC_UPDATE_ITERS,
    tol: float = SSC_TOL,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
):
    encoder.train()
    N = data.x.size(0)

    mu = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32, device=device))
    opt = torch.optim.SGD(
        list(encoder.parameters()) + [mu],
        lr=lr,
        momentum=SSC_SGD_MOMENTUM,
        weight_decay=SSC_SGD_WEIGHT_DECAY,
        nesterov=SSC_SGD_NESTEROV,
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

    def eval_combo(k, n_init, max_iter):
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, init='k-means++', random_state=seed)
        km.fit(embeddings_pre)
        return {"K": int(k), "n_init": int(n_init), "max_iter": int(max_iter), "inertia": float(km.inertia_)}

    if K_TUNE:
        combos = [(k, n_init, max_iter)
                  for k in range(K_MIN, K_MAX + 1)
                  for n_init in KMEANS_N_INIT_GRID
                  for max_iter in KMEANS_MAX_ITER_GRID]
        results = Parallel(n_jobs=K_TUNE_JOBS, verbose=0)(
            delayed(eval_combo)(k, n_init, max_iter) for (k, n_init, max_iter) in combos
        )
        df_k = pd.DataFrame(results).sort_values(["inertia", "K"], ascending=[True, True], ignore_index=True)

        df_elbow = (
            df_k.groupby("K", as_index=False)["inertia"]
            .min()
            .sort_values("K"))
        plt.figure(figsize=(6, 4))
        plt.plot(df_elbow["K"], df_elbow["inertia"], marker="o")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method (K vs Inertia)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        best_row = df_k.iloc[0].to_dict() # select index based on K
        best_k = int(best_row["K"])
        best_n_init = int(best_row["n_init"])
        best_max_iter = int(best_row["max_iter"])
        best_inertia = float(best_row["inertia"])
        km_best = KMeans(n_clusters=best_k, n_init=best_n_init, max_iter=best_max_iter, init='k-means++', random_state=seed)
        y_pred = km_best.fit_predict(embeddings_pre)
    else:
        best_k = int(K_CLUSTERS)
        best_n_init = int(KMEANS_N_INIT_GRID[0])
        best_max_iter = int(KMEANS_MAX_ITER_GRID[0])
        km_best = KMeans(n_clusters=best_k, n_init=best_n_init, max_iter=best_max_iter, init='k-means++', random_state=seed)
        y_pred = km_best.fit_predict(embeddings_pre)
        best_inertia = float(km_best.inertia_)


    y_pred_initial = y_pred.copy()
    q_soft = None
    ssc_hist = None

    if SELF_TRAIN:
        init_centers = km_best.cluster_centers_.astype(np.float32)
        y_refined, centers_refined, ssc_details = ssc_refine(
            encoder=encoder, data=data, init_centers=init_centers,
            alpha=SSC_ALPHA, lr=SSC_LR, max_epochs=SSC_MAX_EPOCHS,
            update_interval=SSC_UPDATE_ITERS, tol=SSC_TOL,
            batch_size=cfg["SSC_BATCHSIZE"], device=device,
        )
        y_pred = y_refined
        q_soft = ssc_details.get("q", None)
        ssc_hist = ssc_details
    encoder.eval()
    with torch.no_grad():
        z_final, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
    embeddings_final = z_final.detach().cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

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
            df_k.to_csv(os.path.join(run_dir, "kmeans_tuning.csv"), index=False)

        np.save(os.path.join(run_dir, "embeddings_pre_ssc.npy"), embeddings_pre)
        np.save(os.path.join(run_dir, "embeddings_post_ssc.npy"), embeddings_final)

        pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv(os.path.join(run_dir, "kmeans_labels_final.csv"), index=False)

        summary = {
            "ARI": float(ari), "NMI": float(nmi),
            "BEST_K": int(best_k), "BEST_N_INIT": int(best_n_init), "BEST_MAX_ITER": int(best_max_iter),
            "BEST_INERTIA": float(best_inertia),
            "SELF_TRAIN": bool(SELF_TRAIN),
            "SSC_ALPHA": float(SSC_ALPHA),
            "SSC_TOL": float(SSC_TOL),
            "GPU_ID": gpu_id if gpu_id is not None else -1,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"cfg": cfg, "summary": summary}, f, indent=2)

        if SELF_TRAIN:
            if q_soft is not None:
                np.save(os.path.join(run_dir, "ssc_Q_soft.npy"), q_soft)
            if ssc_hist is not None:
                with open(os.path.join(run_dir, "ssc_history.json"), "w") as f:
                    json.dump({
                        "changed_frac": ssc_hist.get("changed_frac", []),
                        "kl": ssc_hist.get("kl", []),
                        "epochs_ran": ssc_hist.get("epochs_ran", 0)
                    }, f, indent=2)
            pd.DataFrame({
                "true": y_true,
                "kmeans_init": y_pred_initial,
                "refined": y_pred
            }).to_csv(os.path.join(run_dir, "ssc_labels_refined.csv"), index=False)

    return {
        "cfg": cfg,
        "best_k": int(best_k),
        "best_n_init": int(best_n_init),
        "best_max_iter": int(best_max_iter),
        "best_inertia": float(best_inertia),
        "ari": float(ari),
        "nmi": float(nmi)
    }


def expand_grid(param_grid: dict):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))

def main():
    NUM_REPEATS = 10
    assert NUM_REPEATS == len(REP_SEEDS)

    out_dir = "res_grid"
    os.makedirs(out_dir, exist_ok=True)

    gpus = get_devices_for_workers()
    if not gpus:
        print("No CUDA device found, running on CPU.")
    num_gpus = max(1, len(gpus))

    n_jobs = num_gpus if N_JOBS in (-1, 0) else max(1, N_JOBS)
    print(f"Using {n_jobs} parallel workers across {num_gpus} GPUs: {gpus}")

    all_ari, all_nmi = [], []

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
                "ARI": r["ari"],
                "NMI": r["nmi"],
                "BEST_K": r["best_k"],
                "BEST_N_INIT": r["best_n_init"],
                "BEST_MAX_ITER": r["best_max_iter"],
                "BEST_INERTIA": r["best_inertia"],
            }
            for r in results
        ]).sort_values(["ARI", "NMI"], ascending=False, ignore_index=True)

        df.to_csv(os.path.join(rep_dir, "grid_results.csv"), index=False)

        best = df.iloc[0]
        all_ari.append(best["ARI"])
        all_nmi.append(best["NMI"])

        print(
            f"Rep {rep+1}: ARI={best['ARI']:.4f}, NMI={best['NMI']:.4f}"
        )

    print("\n=== SUMMARY OF 10 RUNS ===")
    print("ARI mean/std:", np.mean(all_ari), np.std(all_ari))
    print("NMI mean/std:", np.mean(all_nmi), np.std(all_nmi))
    
    summary_df = pd.DataFrame({
        "ARI": all_ari,
        "NMI": all_nmi
    })
    summary_df.loc["mean"] = summary_df.mean(numeric_only=True)
    summary_df.loc["std"] = summary_df.std(numeric_only=True)
    summary_df.to_csv(os.path.join(out_dir, "summary_10runs.csv"))

if __name__ == "__main__":
    main()
