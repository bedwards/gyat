#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('nvcc --version')
# get_ipython().system('nvidia-smi')


# In[ ]:


# get_ipython().system('pip3 install torch_geometric')


# In[ ]:


import warnings
import gc
from glob import glob
from functools import partial
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch_geometric.utils import subgraph
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

print(f"torch : {torch.__version__}")

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.min_rows", 10)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", None)

sns.set_theme(style="whitegrid")

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
project_path = "gyat/input"

try:
  from google.colab import drive

except ImportError:
  # Google Cloud - Compute Engine - virtual machine instance
  #
  # ssh with port forwarding
  #   gcloud compute ssh gyat -- -L 8888:localhost:8888
  #
  # mount Google Drive with rclone (https://rclone.org/)
  #   rclone mount gdrive: ~/google-drive --vfs-cache-mode full --vfs-cache-max-size 5G --buffer-size 256M --transfers 4 --daemon
  #
  # Copy files to local storage with progress
  #   rsync -avh --progress ~/google-drive/Colab\ Notebooks/gyat/input/gyat-dataset/*.csv ~/local-data/gyat/input/gyat-dataset/
  #
  # Start jupyter notebook server and copy 127.0.0.1 URL with token
  #   jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --ip=0.0.0.0 --NotebookApp.port_retries=0
  #
  # Google Colab - Connect to a local runtime (paste 127.0.0.1 URL with token from jupyter notebook command)
  #
  assert device == "cuda:0", device
  drive_path = "/home/bedwards/local-data"
  base_path = f"{drive_path}/{project_path}"

else:
  # Google Colab hosted runtime
  drive_path = "/content/drive"
  drive.mount(drive_path)
  base_path = f"{drive_path}/Colab Notebooks/My Drive/{project_path}"

data_path = f"{base_path}/march-machine-learning-mania-2025"
gyat_path = f"{base_path}/gyat-dataset"
state_dict_path = f"{base_path}/state-dicts"
n_csv_files = len(glob(f'{gyat_path}/*.csv'))
print(f"data  : {n_csv_files} csv files at {gyat_path}")
assert n_csv_files > 0, f"no csv files found at {gyat_path}"


# In[ ]:


def print_df(name, df, info=False):
  print(f"{name} {df.shape}")
  print(df)
  print()
  if info:
    df.info()
    print()


# In[ ]:


def load_nodes(path):
  nodes = pd.read_csv(path)
  nodes["Date"] = pd.to_datetime(nodes["Date"])

  nodes = pd.concat([
      # indentifying info, not passed to model
      nodes[["Index"]].astype("int32"),
      nodes[["Key"]],
      nodes[["Season"]].astype("int32"),
      nodes[["Date"]],
      nodes[["Le_TeamID", "Ri_TeamID"]].astype("int32"),
      nodes[["Le_TeamName", "Ri_TeamName"]],

      # target (scaled as Le_y)
      nodes[["Le_Margin"]].astype("int32"),

      # features (not scaled)
      nodes[["Men", "NCAATourney", "Le_Loc"]].astype("float32"),

      # features (scaled)
      nodes.loc[:, "SeasonsAgo":].astype("float32"),
    ],
    axis=1,
  )

  nodes.index = nodes.index.astype("int32")

  return nodes


# In[ ]:


def load_edges(path):
  edges = pd.read_csv(path, dtype="int32")
  edges = edges[edges["Type"] < 4]
  edges[["Direction", "Delta"]] = edges[["Direction", "Delta"]].astype("float32")
  edges.index = edges.index.astype("int32")
  return edges


# In[ ]:


class Graph:
  def __init__(self, season, gender, order):
    self.season = season
    self.gender = gender
    self.order = order
    self.nodes = None
    self.sea = None
    self.edges = None
    self.train = None

  def __repr__(self):
    n = "n?" if self.nodes is None else self.nodes.shape[0]
    s = "s?" if self.sea is None else self.sea.shape[0]
    e = "e?" if self.edges is None else self.edges.shape[0]
    return f"{self.season} {self.gender} {self.order} {n} {s} {e}"


graphs = {}

for path in glob(f"{gyat_path}/*.csv"):
  season, gender, type_ = path.split("/")[-1].split(".")[0].split("_", 2)
  season = int(season)

  if (season, gender, "asc") not in graphs:
    graphs[(season, gender, "asc")] = Graph(season, gender, "asc")

  asc = graphs[(season, gender, "asc")]

  if (season, gender, "des") not in graphs:
    graphs[(season, gender, "des")] = Graph(season, gender, "des")

  des = graphs[(season, gender, "des")]

  if type_ == "edges":
    edges = load_edges(path)
    asc.edges = edges
    des.edges = edges
    continue

  graph = asc if type_.endswith("asc") else des

  if type_.startswith("nodes"):
    graph.nodes = load_nodes(path)
    continue

  graph.sea = load_nodes(path)

graphs = {k: v for k, v in sorted(graphs.items())}

for key, graph in graphs.items():
  print(f"{key}: {graph}")


# In[ ]:


for key, graph in graphs.items():
  if key[2] == "des":
    continue
  asc = graph
  des = graphs[(key[0], key[1], "des")]
  print(f"{asc}\n")
  print(f"{asc.nodes}\n")
  print(f"{asc.sea}\n")
  print("\n")
  print(f"{des}\n")
  print(f"{des.nodes}\n")
  print(f"{des.sea}")
  break


# In[ ]:


def scale(scaler, df, cols=None):
  return pd.DataFrame(
    scaler.transform(df).astype("float32"),
    index=df.index,
    columns=df.columns if cols is None else cols,
  )


scaler_x = StandardScaler()
scaler_y = StandardScaler()


def scale_values(nodes):
  return pd.concat([
      nodes.loc[:, :"Le_Margin"],
      scale(scaler_y, nodes[["Le_Margin"]], ["Le_y"]),
      nodes.loc[:, "Men":"Le_Loc"].astype("float32"),
      scale(scaler_x, nodes.loc[:, "SeasonsAgo":]),
    ],
    axis=1,
  )


# In[ ]:


nodes = pd.concat([graph.nodes for graph in graphs.values()])
scaler_x.fit(nodes.loc[:, "SeasonsAgo":])
scaler_y.fit(nodes[["Le_Margin"]])

for graph in graphs.values():
  graph.nodes = scale_values(graph.nodes)
  graph.sea = scale_values(graph.sea)

print(f"{graphs[(2003, 'Men', 'asc')].nodes}")
print(f"{graphs[(2003, 'Men', 'asc')].sea}")


# In[ ]:


def to_edge_index(edges):
  return torch.tensor(
    edges[["SourceIndex", "TargetIndex"]].T.values,
    device="cpu",
    dtype=torch.long
  )


for (season, gender, order), graph in graphs.items():
  if season < 2021 or season > 2024:
    graph.train = (graph.nodes, graph.edges)
    continue

  reg_season = graph.nodes[graph.nodes["NCAATourney"] == 0]
  ei = to_edge_index(graph.edges)

  _, _, edge_mask = subgraph(
    reg_season.index.to_list(),
    ei,
    return_edge_mask=True,
  )

  graph.train = (
    reg_season,
    graph.edges[edge_mask.numpy()],
  )

  del ei, edge_mask
  torch.cuda.empty_cache()
  gc.collect()


# In[ ]:


print(f"{graphs[(2021, 'Men', 'asc')].nodes}")
print(f"{graphs[(2021, 'Men', 'asc')].train[0]}")
print()
print()
print()
print(f"{graphs[(2021, 'Men', 'asc')].edges}")
print(f"{graphs[(2021, 'Men', 'asc')].train[1]}")


# In[ ]:


def tensor(data):
  return torch.tensor(data.values, device=device, dtype=torch.float32)


def long_tensor(data):
  return torch.tensor(data.values, device=device, dtype=torch.long)


class Model(nn.Module):
  def __init__(self, layers, transforms):
    super().__init__()
    self.layers = nn.ModuleList(layers)
    self.transforms = transforms

  def forward(self, node_indices, x, edge_index, edge_type, edge_attr):
    y_pred = x[node_indices]

    ei, _, mask = subgraph(
        node_indices,
        edge_index,
        relabel_nodes=True,
        return_edge_mask=True,
    )

    for transform in self.transforms:
      y_pred = transform(
        y_pred,
        ei,
        edge_type[mask],
        edge_attr[mask],
      )

    return y_pred


def transform_rgat(layer, x, *edge_args):
  edge_index, edge_type, edge_attr = edge_args
  # print(
  #   f"    Processing graph:\n"
  #   f"      x          {tuple(x.size())}\n"
  #   f"      edge_index {tuple(edge_index.size())}\n"
  #   f"      edge_type  {tuple(edge_type.size())}\n"
  #   f"      edge_attr  {tuple(edge_attr.size())}\n"
  # )

  out = layer(x, *edge_args)
  out = F.leaky_relu(out)
  return F.dropout(out, training=layer.training)


def transform_linear(layer, x, *edge_args):
  return layer(x)


def initialize_model(layer_sizes, heads):
  layers = []
  transforms = []

  for i in range(len(layer_sizes) - 1):
    inp = layer_sizes[i] * (heads if i > 0 else 1)
    out = layer_sizes[i + 1]

    if i < len(layer_sizes) - 2:
      layer = RGATConv(
        inp,
        out,
        num_relations=edges["Type"].unique().shape[0],
        heads=heads,
        edge_dim=len(["Direction", "Delta"]),
      )

      transform = partial(transform_rgat, layer)

    else:
      layer = nn.Linear(inp, out)
      transform = partial(transform_linear, layer)

    layers.append(layer)
    transforms.append(transform)

  model = Model(layers, transforms)
  model.to(device)
  return model


def brier_score(y_pred, margin_true):
    margin_pred = scaler_y.inverse_transform(
      y_pred.cpu().numpy().reshape(-1, 1)
    ).flatten()

    win_prob_pred = 1 / (1 + np.exp(-margin_pred * 0.175))
    win_true = (margin_true > 0).astype("int32")
    return np.mean((win_prob_pred - win_true)**2)


# In[ ]:


class FoldModel:
  def __init__(self, layer_sizes, heads, state_dict=None):
    self.model = initialize_model(layer_sizes, heads)
    self.model.to(device)
    if state_dict is not None:
      self.model.load_state_dict(state_dict)
    self.optimizer = torch.optim.Adam(self.model.parameters())
    self.best_loss = float("inf")
    self.patience_count = 0
    self.best_state_dict = self.model.state_dict()
    self.fold_indices = {}

  def update_best(self, loss):
    if loss < self.best_loss:
      self.best_loss = loss
      self.patience_count = 0
      self.best_state_dict = self.model.state_dict()
      return True
    else:
      self.patience_count += 1
      return False


# In[ ]:


def create_fold_indices(kfold, graphs, fold_models):
  print("  ", end="")

  for key, graph in graphs.items():
    print(f"{key}", sep=", ", end="")
    indices = np.arange(len(graph.train[0]))

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(indices), 1):
      fold_models[fold_n - 1].fold_indices[key] = (i_fold, i_oof)

  print()
  return fold_models


def iterate_over_epochs(n_epochs, patience, graphs, fold_models):
  for epoch_n in range(1, n_epochs + 1):
    print(f"  epoch {epoch_n}")
    epoch_start = datetime.now()
    all_folds_done = True  # default if all patience has expired

    for fold_n, fold_model in enumerate(fold_models, 1):
      print(f"    fold {fold_n}")
      if fold_model.patience_count > patience:
        continue

      all_folds_done = False
      fold_start = datetime.now()
      fold_losses = []
      oof_losses = []
      print("      ", end="")

      for graph_n, (key, graph) in enumerate(graphs.items(), 1):
        if graph_n < 4 or graph_n > (len(graphs) - 4):
          print(f"{key}", sep=", ", end="", flush=True)
        else:
          print(".", end="", flush=True)

        nodes, edges = graph.train
        i_fold, i_oof = fold_model.fold_indices[key]
        i_fold = long_tensor(nodes.index[i_fold])
        i_oof = long_tensor(nodes.index[i_oof])
        x = tensor(nodes.loc[:, "Men":])
        y_true = tensor(nodes[["Le_y"]])
        edge_index = long_tensor(edges[["SourceIndex", "TargetIndex"]].T)
        edge_type = long_tensor(edges["Type"])
        edge_attr = tensor(edges[["Direction", "Delta"]])
        fold_model.model.train()
        y_pred_train = fold_model.model(i_fold, x, edge_index, edge_type, edge_attr)
        mse_train = F.mse_loss(y_pred_train, y_true[i_fold])
        fold_model.optimizer.zero_grad()
        mse_train.backward()
        fold_model.optimizer.step()
        fold_losses.append(mse_train.item())
        fold_model.model.eval()

        with torch.no_grad():
          y_pred_valid = fold_model.model(i_oof, x, edge_index, edge_type, edge_attr)
          mse_valid = F.mse_loss(y_pred_valid, y_true[i_oof])
          oof_losses.append(mse_valid.item())

        del x, y_true, edge_index, edge_type, edge_attr, i_fold, i_oof
        torch.cuda.empty_cache()
        gc.collect()

      print()
      avg_fold_loss = sum(fold_losses) / len(fold_losses) if fold_losses else float("inf")
      avg_oof_loss = sum(oof_losses) / len(oof_losses) if oof_losses else float("inf")
      # print(f"      fold {fold_n}: in-fold loss={avg_fold_loss:.4f}, oof loss={avg_oof_loss:.4f}")
      is_best = fold_model.update_best(avg_oof_loss)

      if is_best:
        torch.save(fold_model.best_state_dict, f"{state_dict_path}/fold_{fold_n}.pt")

      if True or (epoch_n % (n_epochs // 100) == 0 or
          epoch_n > (n_epochs - 3) or
          fold_model.patience_count > patience - 5):
        fold_time = (datetime.now() - fold_start).total_seconds()
        print(
          f"    epoch {epoch_n:>6}, fold {fold_n}: "
          f"in-fold={avg_fold_loss:.4f} "
          f"oof={avg_oof_loss:.4f} "
          f"patience={fold_model.patience_count}/{patience} "
          f"time={fold_time:.1f}s"
        )

      if fold_model.patience_count > patience:
        print(f"    fold {fold_n} out of patience: valid={fold_model.best_loss:.4f}")

    if all_folds_done:
      print(f"All folds done at epoch {epoch_n}")
      break

    epoch_time = (datetime.now() - epoch_start).total_seconds()
    if True or epoch_n % (n_epochs // 100) == 0:
      print(f"  epoch {epoch_n} time: {epoch_time:.1f}s")

  return fold_models


def calculate_oof_predictions_and_scores(graphs, fold_models):
  all_scores = {}

  for key, graph in graphs.items():
    print(f"  {key}")
    nodes, edges = graph.train
    x = tensor(nodes.loc[:, "Men":])
    edge_index = long_tensor(edges[["SourceIndex", "TargetIndex"]].T)
    edge_type = long_tensor(edges["Type"])
    edge_attr = tensor(edges[["Direction", "Delta"]])
    y_pred_oof = np.zeros(len(nodes))
    mask = np.zeros(len(nodes), dtype=bool)

    for fold_n, fold_model in enumerate(fold_models, 1):
      fold_model.model.load_state_dict(fold_model.best_state_dict)
      fold_model.model.eval()
      _, i_oof = fold_model.fold_indices[key]
      i_oof = long_tensor(nodes.index[i_oof])

      with torch.no_grad():
        y_pred = fold_model.model(i_oof, x, edge_index, edge_type, edge_attr).flatten().cpu().numpy()

        for i, pos in enumerate(i_oof):
          y_pred_oof[pos] = y_pred[i]
          mask[pos] = True

    assert mask.all(), "Not all nodes have predictions"

    y_pred_oof_tensor = torch.tensor(y_pred_oof, device=device, dtype=torch.float32)
    score = brier_score(y_pred_oof_tensor, nodes["Le_Margin"])
    all_scores[key] = score
    print(f"    oof brier score: {score:.4f}")
    del x, edge_index, edge_type, edge_attr, y_pred_oof, y_pred_oof_tensor, mask
    torch.cuda.empty_cache()
    gc.collect()

  return all_scores


def train_kfold_models(
    kfold,
    layer_sizes,
    heads,
    state_dicts=None,
    n_epochs=10_000,
    patience=60,
  ):
  fold_models = []

  for i in range(kfold.n_splits):
    state_dict = None if state_dicts is None else state_dicts[i]
    fold_models.append(FoldModel(layer_sizes, heads, state_dict))

  print("Creating fold indices...")
  fold_models = create_fold_indices(kfold, graphs, fold_models)
  print("Iterating over epochs...")
  fold_models = iterate_over_epochs(n_epochs, patience, graphs, fold_models)
  print("Calculating OOF predictions and scores...")
  all_scores = calculate_oof_predictions_and_scores(graphs, fold_models)
  avg_score = sum(all_scores.values()) / len(all_scores)
  print(f"Average OOF Brier score: {avg_score:.4f}")
  return [fold_model.best_state_dict for fold_model in fold_models]


# In[ ]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
n_features = list(graphs.values())[0].train[0].loc[:, "Men":].shape[1]
layer_sizes = [n_features, 16, 8, 1]
heads = 2
state_dicts = train_kfold_models(kfold, layer_sizes, heads)


# In[ ]:


# def test_models(layer_sizes, state_dicts):
#   y_preds = [
#     torch.zeros(y_true.shape[0], device=device, dtype=torch.float32)
#     for y_true in y_trues
#   ]

#   for state_dict in state_dicts:
#     model = initialize_model(layer_sizes)
#     model.load_state_dict(state_dict)
#     model.eval()

#     with torch.no_grad():
#       for x, y_pred in zip(xs, y_preds):
#         y_pred += model.forward(long_tensor(test), x).flatten()

#   for y_pred in y_preds:
#     y_pred /= len(state_dicts)

#   score = calculate_score(y_preds, test)
#   print(f"test brier score: {score:.4f}")


# In[ ]:


# test_models(layer_sizes, state_dicts)


# In[ ]:




