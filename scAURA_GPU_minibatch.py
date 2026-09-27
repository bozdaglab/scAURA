import os, json, math, contextlib, random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
from itertools import product
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import GCL.augmentors as A
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from scipy.sparse import coo_matrix
from pathlib import Path



SEED = 42
K_CLUSTERS = 55
K_TUNE = True
K_MIN = 50
K_MAX = 55
DEC_INIT_METHODS = ["kmeans", "spectral"]
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 300
SILHOUETTE_SAMPLE_SIZE = 10000
SPECTRAL_AFFINITY = "nearest_neighbors"
SPECTRAL_N_NEIGHBORS = 20
SPECTRAL_ASSIGN_LABELS = "discretize"



SELF_TRAIN = True

X_PATH = Path(__file__).parent / "dataset" / "Tabula_Muris" / "y_hvg_500.npy"
Y_PATH = Path(__file__).parent / "dataset" / "Tabula_Muris" / "y_hvg_500.npy"

PARAM_GRID = {
    "USE_PCA_FOR_GRAPH": [True],
    "PCA_N_PCS": [40],
    "AK_KMAX": [40],
    "AK_DELTA": [2],
    "GRAPH_METRIC": ["euclidean", "cosine"],

    "HIDDEN_DIM": [64],
    "PROJ_DIM": [32],
    "NUM_GCN_LAYERS": [2],
    "ACTIVATION": ["tanh", "relu", "sigmoid"],

    "PE": [0.1],
    "PF": [0.1],
    "TAU": [0.5],
    "TAU_PLUS": [0.1],
    "LR": [1e-3],
    "EPOCHS": [200],

    "LAMBDA_ALIGN": [1.0],
    "LAMBDA_UNIFORM": [1.0],
    "UNIFORM_WARMUP": [20],
    "NORMALIZE_EMBEDS": [True],


    "TRAIN_BATCH_SIZE": [256],
    "EMBED_BATCH_SIZE": [512],
    "NUM_NEIGHBORS": [[5, 5]],
    "NUM_WORKERS": [0],
    "USE_AMP": [False],
    "EDGE_AUGMENT": [False],
    "GRAD_ACCUM_STEPS": [4],
}

REP_SEEDS = [101, 370, 124, 319, 2024, 8080, 4142, 141, 371, 421]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@contextlib.contextmanager
def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        except AttributeError:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                yield
    else:
        yield


def get_devices() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def pick_device(gpu_id: Optional[int]) -> torch.device:
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, ei1, ew1)
        z2 = self.encoder(x2, ei2, ew2)
        return z, z1, z2


    def encode(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)

    def encode_views(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        if aug1 is not None and aug2 is not None:
            x1, ei1, ew1 = aug1(x, edge_index, edge_weight)
            x2, ei2, ew2 = aug2(x, edge_index, edge_weight)
        else:

            x1, x2 = x, x
            ei1, ei2 = edge_index, edge_index
            ew1, ew2 = edge_weight, edge_weight
        z1 = self.encoder(x1, ei1, ew1)
        z2 = self.encoder(x2, ei2, ew2)
        return z1, z2

    def project(self, z):
        return self.fc2(F.elu(self.fc1(z)))


def get_act(name: str):
    lut = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    name = name.lower()
    if name not in lut:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(lut)}")
    return lut[name]



def _choose_adaptive_k(d_row_sorted, Kmax, delta):
    denom = max(1e-8, (Kmax - 1 - delta))
    T = (np.sum(np.sqrt(d_row_sorted)) / denom) ** 2
    k = np.searchsorted(d_row_sorted, T, side="right")
    return max(1, int(k))


def _build_adaptive_knn(X, Kmax, delta, use_pca, n_pcs, metric, random_state=SEED):
    if use_pca:
        pca = PCA(n_components=min(n_pcs, X.shape[1]), random_state=random_state)
        X_ = pca.fit_transform(X)
    else:
        X_ = X

    nnbr = NearestNeighbors(n_neighbors=Kmax, metric=metric, n_jobs=-1)
    nnbr.fit(X_)
    dists, nbrs = nnbr.kneighbors(X_)

    N = X.shape[0]
    ks = np.array([
        _choose_adaptive_k(np.sort(dists[i]), Kmax=Kmax, delta=delta)
        for i in range(N)
    ], dtype=np.int32)
    rows = np.repeat(np.arange(N), ks)
    cols = np.concatenate([nbrs[i, :ks[i]] for i in range(N)])
    return rows, cols, nbrs, ks


def _snn_jaccard(rows, cols, nbrs, ks, symmetric=True):
    N = nbrs.shape[0]
    neigh_sets = [set(nbrs[i, :ks[i]].tolist()) for i in range(N)]
    weights = np.empty(rows.shape[0], dtype=np.float32)

    for idx, (i, j) in enumerate(zip(rows, cols)):
        Ni, Nj = neigh_sets[i], neigh_sets[j]
        inter = len(Ni & Nj)
        uni = len(Ni | Nj) if (Ni or Nj) else 1
        weights[idx] = inter / max(1, uni)

    W = coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()
    if symmetric:
        W = W.maximum(W.T)
    return W


def build_full_graph_cpu(X_np, y_np, cfg, random_state: int) -> Data:
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

    edge_index = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    edge_weight = torch.tensor(W.data, dtype=torch.float32)
    x = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)


    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)



def debiased_infonce_batch(h1n, h2n, tau: float, tau_plus: float):
    """In-batch debiased InfoNCE. Memory is O(B^2), not O(N^2)."""
    B = h1n.size(0)
    M = 2 * B
    emb = torch.cat([h1n, h2n], dim=0)

    sim = (emb @ emb.t()) / tau
    exp_sim = torch.exp(sim).float()

    eye = torch.eye(M, device=emb.device, dtype=torch.bool)
    exp_sim = exp_sim.masked_fill(eye, 0.0)

    pos_idx = torch.arange(M, device=emb.device)
    pos_idx = (pos_idx + B) % M
    pos_exp = exp_sim[torch.arange(M, device=emb.device), pos_idx]


    neg_sum = exp_sim.sum(dim=1) - pos_exp
    Np = max(1, M - 2)
    Ng = (-tau_plus * Np * pos_exp + neg_sum) / max(1e-8, (1.0 - tau_plus))
    Ng = torch.clamp(Ng, min=Np * math.e ** (-1.0 / tau))

    return (-torch.log(pos_exp / (pos_exp + Ng + 1e-12))).mean()


def uniformity_batch(emb, alpha: float = 2.0):
    """Wang-Isola style uniformity computed only within the current batch."""
    pdist2 = torch.pdist(emb.float(), p=2).pow(2)
    if pdist2.numel() == 0:
        return emb.new_tensor(0.0)
    return torch.log(torch.exp(-alpha * pdist2).mean() + 1e-12)


def make_loader(data, batch_size, num_neighbors, shuffle, num_workers):
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )


@torch.no_grad()
def compute_embeddings_batched(encoder, data, cfg, device):
    encoder.eval()
    N = data.num_nodes
    hidden_dim = cfg["HIDDEN_DIM"]
    out = np.empty((N, hidden_dim), dtype=np.float32)

    loader = make_loader(
        data,
        batch_size=cfg["EMBED_BATCH_SIZE"],
        num_neighbors=cfg["NUM_NEIGHBORS"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
    )

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        z = encoder.encode(batch.x, batch.edge_index, batch.edge_attr)
        bs = batch.batch_size
        n_id = batch.n_id[:bs].detach().cpu().numpy()
        out[n_id] = z[:bs].detach().float().cpu().numpy()

    return out


def _silhouette_for_labels(embeddings, labels, seed):
    """Silhouette used consistently for K-means and spectral initialization.
    For large datasets, sample cells to keep the O(N^2) silhouette calculation practical.
    """
    if len(np.unique(labels)) < 2:
        return float("-inf")
    sample_size = SILHOUETTE_SAMPLE_SIZE
    if sample_size is not None and embeddings.shape[0] > sample_size:
        return float(silhouette_score(embeddings, labels, metric="euclidean",
                                      sample_size=sample_size, random_state=seed))
    return float(silhouette_score(embeddings, labels, metric="euclidean"))


def compute_centers_from_labels(embeddings, labels, k):
    centers = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
    global_center = embeddings.mean(axis=0).astype(np.float32)
    for cid in range(k):
        mask = labels == cid
        centers[cid] = embeddings[mask].mean(axis=0) if np.any(mask) else global_center
    return centers


def tune_dec_initialization(embeddings, seed):
    """Test K-means and spectral over the same K range and select max silhouette."""
    ks = range(K_MIN, K_MAX + 1) if K_TUNE else [int(K_CLUSTERS)]
    results = []
    best = None

    for method in DEC_INIT_METHODS:
        for k in ks:
            if method == "kmeans":
                model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                               max_iter=KMEANS_MAX_ITER, init="k-means++",
                               random_state=seed)
                labels = model.fit_predict(embeddings)
            elif method == "spectral":
                n_neighbors = min(SPECTRAL_N_NEIGHBORS, embeddings.shape[0] - 1)
                model = SpectralClustering(
                    n_clusters=k, affinity=SPECTRAL_AFFINITY,
                    n_neighbors=n_neighbors, assign_labels=SPECTRAL_ASSIGN_LABELS,
                    random_state=seed, n_jobs=-1)
                labels = model.fit_predict(embeddings)
            else:
                raise ValueError(f"Unknown DEC initialization method: {method}")

            try:
                sil = _silhouette_for_labels(embeddings, labels, seed)
            except Exception as e:
                print(f"      init={method} K={k}: silhouette failed ({e})", flush=True)
                sil = float("-inf")

            row = {"method": method, "K": int(k), "silhouette": float(sil)}
            results.append(row)
            print(f"      init={method:8s} K={k:3d} silhouette={sil:.6f}", flush=True)
            candidate = {**row, "labels": labels.copy()}
            if best is None or candidate["silhouette"] > best["silhouette"]:
                best = candidate

    if best is None:
        raise RuntimeError("No valid DEC initialization candidate was produced.")
    best["centers"] = compute_centers_from_labels(embeddings, best["labels"], best["K"])
    return best, pd.DataFrame(results).sort_values(
        ["silhouette", "method", "K"], ascending=[False, True, True], ignore_index=True)


def train_one(X_np, y_np, cfg, run_dir=None, save_metrics=True,
              gpu_id: Optional[int] = None, base_seed: Optional[int] = None):
    seed = base_seed if base_seed is not None else SEED
    device = pick_device(gpu_id)
    seed_all(seed)


    data = build_full_graph_cpu(X_np, y_np, cfg, random_state=seed)
    num_features = X_np.shape[1]

    gconv = GConv(
        input_dim=num_features,
        hidden_dim=cfg["HIDDEN_DIM"],
        activation=get_act(cfg["ACTIVATION"]),
        num_layers=cfg["NUM_GCN_LAYERS"],
    ).to(device)

    if cfg.get("EDGE_AUGMENT", False):
        aug1 = A.Compose([A.EdgeRemoving(pe=cfg["PE"]), A.FeatureMasking(pf=cfg["PF"])])
        aug2 = A.Compose([A.EdgeRemoving(pe=cfg["PE"]), A.FeatureMasking(pf=cfg["PF"])])
    else:
        aug1 = A.Compose([A.FeatureMasking(pf=cfg["PF"])])
        aug2 = A.Compose([A.FeatureMasking(pf=cfg["PF"])])
    encoder = Encoder(gconv, (aug1, aug2), hidden_dim=cfg["HIDDEN_DIM"], proj_dim=cfg["PROJ_DIM"]).to(device)
    opt = Adam(encoder.parameters(), lr=cfg["LR"])
    scaler = None

    train_loader = make_loader(
        data,
        batch_size=cfg["TRAIN_BATCH_SIZE"],
        num_neighbors=cfg["NUM_NEIGHBORS"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
    )

    loss_hist, align_hist, uni_hist, tot_hist, wu_hist = [], [], [], [], []

    for epoch in range(cfg["EPOCHS"]):
        encoder.train()
        epoch_contrastive = 0.0
        epoch_align = 0.0
        epoch_uni = 0.0
        epoch_total = 0.0
        nb = 0

        opt.zero_grad(set_to_none=True)
        accum = max(1, int(cfg.get("GRAD_ACCUM_STEPS", 1)))

        for step, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)

            amp_enabled = bool(cfg.get("USE_AMP", False))
            ctx = maybe_autocast(device) if amp_enabled else contextlib.nullcontext()
            with ctx:
                z1, z2 = encoder.encode_views(batch.x, batch.edge_index, batch.edge_attr)



                bs = batch.batch_size
                h1 = encoder.project(z1[:bs])
                h2 = encoder.project(z2[:bs])

                if cfg["NORMALIZE_EMBEDS"]:
                    h1n, h2n = F.normalize(h1, dim=1), F.normalize(h2, dim=1)
                else:
                    h1n, h2n = h1, h2

                align = (h1n - h2n).pow(2).sum(dim=1).mean()
                contrastive = debiased_infonce_batch(
                    h1n, h2n,
                    tau=cfg["TAU"],
                    tau_plus=cfg["TAU_PLUS"],
                )
                emb_full = torch.cat([h1n, h2n], dim=0)
                uniformity = uniformity_batch(emb_full, alpha=2.0)

                wu = min(1.0, (epoch + 1) / float(max(1, cfg["UNIFORM_WARMUP"]))) \
                    if cfg["UNIFORM_WARMUP"] > 0 else 1.0
                total_loss = (
                    contrastive
                    + cfg["LAMBDA_ALIGN"] * align
                    + (cfg["LAMBDA_UNIFORM"] * wu) * uniformity
                )
                loss_for_backward = total_loss / accum

            loss_for_backward.backward()
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()


            epoch_contrastive += float(contrastive.detach().cpu())
            epoch_align += float(align.detach().cpu())
            epoch_uni += float(uniformity.detach().cpu())
            epoch_total += float(total_loss.detach().cpu())
            nb += 1

        if nb % max(1, int(cfg.get("GRAD_ACCUM_STEPS", 1))) != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        loss_hist.append(epoch_contrastive / max(1, nb))
        align_hist.append(epoch_align / max(1, nb))
        uni_hist.append(epoch_uni / max(1, nb))
        tot_hist.append(epoch_total / max(1, nb))
        wu_hist.append(float(wu))

        print(
            f"      epoch {epoch + 1:03d}/{cfg['EPOCHS']} "
            f"loss={loss_hist[-1]:.4f} align={align_hist[-1]:.4f} "
            f"uni={uni_hist[-1]:.4f} total={tot_hist[-1]:.4f}",
            flush=True,
        )

    embeddings = compute_embeddings_batched(encoder, data, cfg, device)

    best_init, init_tuning_df = tune_dec_initialization(embeddings, seed)
    best_k = int(best_init["K"])
    best_init_method = str(best_init["method"])
    best_init_silhouette = float(best_init["silhouette"])
    y_pred = best_init["labels"]

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
            "uniformity_weight": wu_hist,
        }).to_csv(os.path.join(run_dir, "training_metrics.csv"), index=False)

        init_tuning_df.to_csv(os.path.join(run_dir, "dec_init_tuning.csv"), index=False)
        np.save(os.path.join(run_dir, "embeddings.npy"), embeddings)
        pd.DataFrame({"true": y_np, "pred": y_pred}).to_csv(
            os.path.join(run_dir, "labels_final.csv"), index=False)

        summary = {
            "ARI": float(ari),
            "NMI": float(nmi),
            "BEST_K": best_k,
            "BEST_DEC_INIT_METHOD": best_init_method,
            "BEST_INIT_SILHOUETTE": best_init_silhouette,
            "SELF_TRAIN": SELF_TRAIN,
            "CLUSTERING": best_init_method,
            "SPECTRAL_AFFINITY": SPECTRAL_AFFINITY,
            "SPECTRAL_N_NEIGHBORS": SPECTRAL_N_NEIGHBORS,
            "SPECTRAL_ASSIGN_LABELS": SPECTRAL_ASSIGN_LABELS,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"cfg": cfg, "summary": summary}, f, indent=2)

    return {
        "cfg": cfg,
        "best_k": best_k,
        "best_dec_init_method": best_init_method,
        "best_init_silhouette": best_init_silhouette,
        "ari": float(ari),
        "nmi": float(nmi),
    }


def expand_grid(param_grid: dict):
    keys = list(param_grid.keys())
    for combo in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def main():
    NUM_REPEATS = 10
    assert NUM_REPEATS == len(REP_SEEDS), \
        f"NUM_REPEATS={NUM_REPEATS} but REP_SEEDS has {len(REP_SEEDS)} entries"

    print(f"Loading data from {X_PATH} and {Y_PATH} ...")
    X_np = np.load(X_PATH)
    y_np = np.load(Y_PATH)
    print(f"X shape: {X_np.shape},  y shape: {y_np.shape}")
    print(f"Unique labels: {np.unique(y_np).shape[0]}")

    out_dir = "res_grid_minibatch"
    os.makedirs(out_dir, exist_ok=True)

    gpus = get_devices()
    num_gpus = max(1, len(gpus))
    print(f"GPUs available: {gpus}")

    grid = list(expand_grid(PARAM_GRID))
    print(f"Grid size: {len(grid)} configs × {NUM_REPEATS} repeats")

    all_ari, all_nmi = [], []

    for rep in range(NUM_REPEATS):
        rep_seed = REP_SEEDS[rep]
        rep_name = f"rep_{rep:02d}"
        print(f"\n=== RUN {rep + 1}/{NUM_REPEATS}  (seed={rep_seed}) ===")

        run_results = []

        for i, cfg in enumerate(grid):
            gpu_id = gpus[i % num_gpus] if gpus else None



            exp_dir = os.path.join(out_dir, rep_name, f"exp_{i:04d}")
            os.makedirs(exp_dir, exist_ok=True)

            print(f"  exp {i + 1}/{len(grid)}  gpu={gpu_id}  cfg={cfg}")

            try:
                res = train_one(
                    X_np, y_np, cfg,
                    run_dir=exp_dir,
                    save_metrics=True,
                    gpu_id=gpu_id,
                    base_seed=rep_seed,
                )

                row = {
                    "rep": rep,
                    "run": rep + 1,
                    "seed": rep_seed,
                    "exp": i,
                    "ARI": res["ari"],
                    "NMI": res["nmi"],
                    "best_k": res["best_k"],
                    "best_dec_init_method": res["best_dec_init_method"],
                    "best_init_silhouette": res["best_init_silhouette"],
                }
                row.update(cfg)
                run_results.append(row)

                print(f"    ARI={res['ari']:.4f}  NMI={res['nmi']:.4f}")

            except torch.cuda.OutOfMemoryError as e:
                print(f"    OOM on exp {i} — skipping. ({e})")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if len(run_results) == 0:
            print(f"Rep {rep + 1}: no successful experiments; writing empty results file.")
            pd.DataFrame().to_csv(
                os.path.join(out_dir, f"{rep_name}_results_sorted.csv"),
                index=False,
            )
            all_ari.append(np.nan)
            all_nmi.append(np.nan)
            continue


        run_df = pd.DataFrame(run_results)
        run_df = run_df.sort_values(["ARI", "NMI"], ascending=False)
        run_df.to_csv(
            os.path.join(out_dir, f"{rep_name}_results_sorted.csv"),
            index=False,
        )

        best_row = run_df.iloc[0]
        all_ari.append(float(best_row["ARI"]))
        all_nmi.append(float(best_row["NMI"]))

        print(
            f"Rep {rep + 1}: best ARI={best_row['ARI']:.4f}  "
            f"NMI={best_row['NMI']:.4f}"
        )


    summary_df = pd.DataFrame({
        "run": np.arange(1, NUM_REPEATS + 1),
        "seed": REP_SEEDS,
        "best_ARI": all_ari,
        "best_NMI": all_nmi,
    })

    ari_mean = np.nanmean(all_ari)
    nmi_mean = np.nanmean(all_nmi)
    ari_std = np.nanstd(all_ari, ddof=1)
    nmi_std = np.nanstd(all_nmi, ddof=1)

    summary_df.loc[len(summary_df)] = ["MEAN", "-", ari_mean, nmi_mean]
    summary_df.loc[len(summary_df)] = ["STD", "-", ari_std, nmi_std]
    summary_df.loc[len(summary_df)] = [
        "MEAN ± STD",
        "-",
        f"{ari_mean:.4f} ± {ari_std:.4f}",
        f"{nmi_mean:.4f} ± {nmi_std:.4f}",
    ]

    summary_df.to_csv(os.path.join(out_dir, "summary_runs.csv"), index=False)

    print("\n=== SUMMARY ===")
    print(f"ARI  {ari_mean:.4f} ± {ari_std:.4f}")
    print(f"NMI  {nmi_mean:.4f} ± {nmi_std:.4f}")

if __name__ == "__main__":
    main()
