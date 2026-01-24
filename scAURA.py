import os, json, warnings
from datetime import datetime
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

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
import random

from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from scipy.sparse import coo_matrix

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
import matplotlib.pyplot as plt
from pathlib import Path



SEED = 42
K_CLUSTERS = 7
N_JOBS = -1
K_TUNE_JOBS = -1

K_TUNE = False
K_MIN = 5
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
    "ACTIVATION":         ['tanh'],

    "PE":                 [0.3],
    "PF":                 [0.3],
    "TAU":                [0.7],
    "TAU_PLUS":           [0.1],
    "LR":                 [1e-3],
    "EPOCHS":             [50, 100, 200],

    "LAMBDA_ALIGN":       [1.0],
    "LAMBDA_UNIFORM":     [1.0],
    "UNIFORM_WARMUP":     [20],
    "NORMALIZE_EMBEDS":   [True],

    "SSC_BATCHSIZE":      [32, 64, 128],
}

REP_SEEDS = [44, 23, 56, 89, 424, 1, 3, 7, 8, 9]



SELF_TRAIN = True
SSC_ALPHA = 1.0
SSC_LR = 1e-3
SSC_MAX_EPOCHS = 500
SSC_UPDATE_ITERS = 10
SSC_TOL = 0.005

SSC_SGD_MOMENTUM = 0.9
SSC_SGD_WEIGHT_DECAY = 0.0
SSC_SGD_NESTEROV = True


torch.manual_seed(SEED);
np.random.seed(SEED)
device = torch.device('cpu')

de_df_temp = pd.read_csv(Path(__file__).parent / "dataset"/"Camp_Liver"/ "camp_liver.csv")
X_np = de_df_temp.iloc[:, 1:501].values.astype(np.float32)
labels = de_df_temp['label']
le = preprocessing.LabelEncoder()
y_np = le.fit_transform(labels).astype(np.int64)
num_features = X_np.shape[1]


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device_ = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device_)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


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

    def project(self, z):
        return self.fc2(F.elu(self.fc1(z)))


def get_act(name: str):
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
        inter = len(Ni & Nj)
        uni = len(Ni | Nj) if (Ni or Nj) else 1
        weights[idx] = inter / max(1, uni)
    W = coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()
    if symmetric:
        W = W.maximum(W.T)
    return W


def build_full_graph(cfg):
    rows, cols, nbrs, ks = _build_adaptive_knn(
        X_np,
        Kmax=cfg["AK_KMAX"],
        delta=cfg["AK_DELTA"],
        use_pca=cfg["USE_PCA_FOR_GRAPH"],
        n_pcs=cfg["PCA_N_PCS"],
        metric=cfg["GRAPH_METRIC"],
        random_state=SEED
    )
    W = _snn_jaccard(rows, cols, nbrs, ks, symmetric=True).tocoo()
    edge_index = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    edge_weight = torch.tensor(W.data, dtype=torch.float32)
    data = Data(
        x=torch.tensor(X_np, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y_np, dtype=torch.long)
    ).to(device)
    return data


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



def train_aura(cfg, run_dir=None, save_metrics=True, base_seed: int = 42):
    seed_all(base_seed)
    data = build_full_graph(cfg)

    gconv = GConv(
        input_dim=num_features,
        hidden_dim=cfg["HIDDEN_DIM"],
        activation=get_act(cfg["ACTIVATION"]),
        num_layers=cfg["NUM_GCN_LAYERS"]
    ).to(device)

    aug1 = Compose([EdgeRemoving(pe=cfg["PE"]), FeatureMasking(pf=cfg["PF"])])
    aug2 = Compose([EdgeRemoving(pe=cfg["PE"]), FeatureMasking(pf=cfg["PF"])])

    encoder = Encoder(gconv, (aug1, aug2), hidden_dim=cfg["HIDDEN_DIM"], proj_dim=cfg["PROJ_DIM"]).to(device)
    opt = Adam(encoder.parameters(), lr=cfg["LR"])

    batch_size = data.x.size(0)
    neg_mask = get_negative_mask(batch_size).to(device)

    loss_hist, align_hist, uni_hist, tot_hist, wu_hist = [], [], [], [], []
    for epoch in range(cfg["EPOCHS"]):
        encoder.train()
        z, z1, z2 = encoder(data.x, data.edge_index, data.edge_attr)
        h1, h2 = encoder.project(z1), encoder.project(z2)

        if cfg["NORMALIZE_EMBEDS"]:
            h1n, h2n = F.normalize(h1, dim=1), F.normalize(h2, dim=1)
        else:
            h1n, h2n = h1, h2

        align = (h1n - h2n).pow(2).sum(dim=1).mean()

        emb = torch.cat([h1n, h2n], dim=0)
        sq = torch.cdist(emb, emb, p=2).pow(2)
        mask = ~torch.eye(sq.size(0), dtype=bool, device=sq.device)
        uniformity = torch.log(torch.exp(-2 * sq[mask]).mean())

        dotmat = torch.mm(emb, emb.t())
        neg_exp = torch.exp(dotmat / cfg["TAU"])
        neg_sel = neg_exp.masked_select(neg_mask)
        neg_exp = neg_sel.view(2 * batch_size, -1)
        pos_exp = torch.exp(torch.sum(h1n * h2n, dim=-1) / cfg["TAU"])
        pos_exp = torch.cat([pos_exp, pos_exp], dim=0)
        Np = batch_size * 2 - 2
        Ng = (-cfg["TAU_PLUS"] * Np * pos_exp + neg_exp.sum(dim=-1)) / (1 - cfg["TAU_PLUS"])
        Ng = torch.clamp(Ng, min=Np * np.e**(-1 / cfg["TAU"]))
        contrastive = (-torch.log(pos_exp / (pos_exp + Ng))).mean()

        wu = min(1.0, (epoch + 1) / float(max(1, cfg["UNIFORM_WARMUP"]))) if cfg["UNIFORM_WARMUP"] > 0 else 1.0
        total_loss = contrastive + cfg["LAMBDA_ALIGN"] * align + (cfg["LAMBDA_UNIFORM"] * wu) * uniformity

        opt.zero_grad(); total_loss.backward(); opt.step()

        loss_hist.append(float(contrastive.item()))
        align_hist.append(float(align.item()))
        uni_hist.append(float(uniformity.item()))
        tot_hist.append(float(total_loss.item()))
        wu_hist.append(float(wu))

    encoder.eval()
    with torch.no_grad():
        z, _, _ = encoder(data.x, data.edge_index, data.edge_attr)
    embeddings = z.cpu().numpy()

    def eval_combo(k, n_init, max_iter):
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, init='k-means++', random_state=SEED)
        km.fit(embeddings)
        return {
            "K": int(k),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "inertia": float(km.inertia_)
        }

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

        km_best = KMeans(n_clusters=best_k, n_init=best_n_init, max_iter=best_max_iter,
                         init='k-means++', random_state=SEED)
        y_pred = km_best.fit_predict(embeddings)
    else:
        best_k = int(K_CLUSTERS)
        best_n_init = int(KMEANS_N_INIT_GRID[0])
        best_max_iter = int(KMEANS_MAX_ITER_GRID[0])
        km_best = KMeans(n_clusters=best_k, n_init=best_n_init, max_iter=best_max_iter,
                         init='k-means++', random_state=SEED)
        y_pred = km_best.fit_predict(embeddings)
        best_inertia = float(km_best.inertia_)

    y_pred_initial = y_pred.copy()
    q_soft = None
    ssc_hist = None

    if SELF_TRAIN:
        init_centers = km_best.cluster_centers_.astype(np.float32)
        y_refined, centers_refined, ssc_details = ssc_refine(
            encoder=encoder,
            data=data,
            init_centers=init_centers,
            alpha=SSC_ALPHA,
            lr=SSC_LR,
            max_epochs=SSC_MAX_EPOCHS,
            update_interval=SSC_UPDATE_ITERS,
            tol=SSC_TOL,
            batch_size=cfg["SSC_BATCHSIZE"],
            device=device,
        )
        y_pred = y_refined
        q_soft = ssc_details.get("q", None)
        ssc_hist = ssc_details

    ari = adjusted_rand_score(y_np, y_pred)
    nmi = normalized_mutual_info_score(y_np, y_pred)

    if save_metrics and run_dir:
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame({
            "epoch": np.arange(1, len(loss_hist) + 1),
            "contrastive_loss": loss_hist,
            "alignment": align_hist,
            "uniformity": uni_hist,
            "total_loss": tot_hist,
            "uniformity_weight": wu_hist
        }).to_csv(os.path.join(run_dir, "training_metrics.csv"), index=False)

        if K_TUNE:
            df_k.to_csv(os.path.join(run_dir, "kmeans_tuning.csv"), index=False)

        np.save(os.path.join(run_dir, "embeddings.npy"), embeddings)
        pd.DataFrame({"true": y_np, "pred": y_pred}).to_csv(os.path.join(run_dir, "kmeans_labels.csv"), index=False)

        summary = {
            "ARI": float(ari),
            "NMI": float(nmi),
            "BEST_K": int(best_k),
            "BEST_N_INIT": int(best_n_init),
            "BEST_MAX_ITER": int(best_max_iter),
            "BEST_INERTIA": float(best_inertia),
            "SELF_TRAIN": bool(SELF_TRAIN),
            "SSC_ALPHA": float(SSC_ALPHA),
            "SSC_TOL": float(SSC_TOL),
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
                "true": y_np,
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
        "nmi": float(nmi),
    }


def expand_grid(param_grid: dict):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    for combo in product(*vals):
        cfg = dict(zip(keys, combo))
        yield cfg


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def main():

    NUM_REPEATS = 10

    out_dir = "res_grid"
    os.makedirs(out_dir, exist_ok=True)

    all_ari, all_nmi = [], []

    grid = list(expand_grid(PARAM_GRID))

    for rep in range(NUM_REPEATS):
        rep_seed = REP_SEEDS[rep]
        print(f"\n=== RUN {rep + 1}/{NUM_REPEATS}  (seed={rep_seed}) ===")

        def run_indexed(idx_cfg):
            idx, cfg = idx_cfg
            run_dir = os.path.join(out_dir, f"rep_{rep:02d}_exp_{idx:04d}")

            res = train_aura(cfg, run_dir=run_dir, save_metrics=True, base_seed=rep_seed)
            res["exp_id"] = idx
            return res

        results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(

            delayed(run_indexed)((i, grid[i])) for i in range(len(grid))
        )

        df = pd.DataFrame([{
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

        df.to_csv(os.path.join(out_dir, f"grid_results_rep{rep:02d}.csv"), index=False)

        best = df.iloc[0]
        all_ari.append(best["ARI"])
        all_nmi.append(best["NMI"])

        print(f"Rep {rep + 1}: ARI={best['ARI']:.4f}, NMI={best['NMI']:.4f}")

    print("\n=== SUMMARY OF 10 RUNS ===")
    print("ARI mean/std:", np.mean(all_ari), np.std(all_ari))
    print("NMI mean/std:", np.mean(all_nmi), np.std(all_nmi))

    summary_df = pd.DataFrame({
        "ARI": all_ari,
        "NMI": all_nmi
    })
    summary_df.loc["mean"] = summary_df.mean()
    summary_df.loc["std"] = summary_df.std()
    summary_df.to_csv(os.path.join(out_dir, "summary_10runs.csv"))


if __name__ == "__main__":
    main()





