```python
!nvcc --version
!nvidia-smi
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Thu_Mar_28_02:18:24_PDT_2024
    Cuda compilation tools, release 12.4, V12.4.131
    Build cuda_12.4.r12.4/compiler.34097967_0
    Mon Apr  7 15:02:03 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA L4                      On  |   00000000:00:03.0 Off |                    0 |
    | N/A   75C    P0             34W /   72W |       1MiB /  23034MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+



```python
!pip3 install torch_geometric
```

    Requirement already satisfied: torch_geometric in /opt/python/3.10/lib/python3.10/site-packages (2.6.1)
    Requirement already satisfied: aiohttp in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (3.9.5)
    Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (2025.3.0)
    Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (3.1.6)
    Requirement already satisfied: numpy in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (1.25.2)
    Requirement already satisfied: psutil>=5.8.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (5.9.3)
    Requirement already satisfied: pyparsing in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (3.2.3)
    Requirement already satisfied: requests in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (2.32.3)
    Requirement already satisfied: tqdm in /opt/python/3.10/lib/python3.10/site-packages (from torch_geometric) (4.67.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (6.2.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.18.3)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /opt/python/3.10/lib/python3.10/site-packages (from aiohttp->torch_geometric) (4.0.3)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch_geometric) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests->torch_geometric) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests->torch_geometric) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests->torch_geometric) (1.26.20)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests->torch_geometric) (2025.1.31)
    Requirement already satisfied: typing-extensions>=4.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from multidict<7.0,>=4.5->aiohttp->torch_geometric) (4.13.0)
    Requirement already satisfied: propcache>=0.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from yarl<2.0,>=1.0->aiohttp->torch_geometric) (0.3.1)



```python
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
n_csv_files = len(glob(f'{gyat_path}/*.csv'))
print(f"data  : {n_csv_files} csv files at {gyat_path}")
assert n_csv_files > 0, f"no csv files found at {gyat_path}"
```

    torch : 2.4.0+cu124
    device: cuda:0
    data  : 195 csv files at /home/bedwards/local-data/gyat/input/gyat-dataset



```python
def print_df(name, df, info=False):
  print(f"{name} {df.shape}")
  print(df)
  print()
  if info:
    df.info()
    print()
```


```python
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
```


```python
def load_edges(path):
  edges = pd.read_csv(path, dtype="int32")
  edges = edges[edges["Type"] < 4]
  edges[["Direction", "Delta"]] = edges[["Direction", "Delta"]].astype("float32")
  edges.index = edges.index.astype("int32")
  return edges
```


```python
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
```

    (2003, 'Men', 'asc'): 2003 Men asc 4680 4680 260196
    (2003, 'Men', 'des'): 2003 Men des 4680 4680 260196
    (2004, 'Men', 'asc'): 2004 Men asc 4635 4635 256206
    (2004, 'Men', 'des'): 2004 Men des 4635 4635 256206
    (2005, 'Men', 'asc'): 2005 Men asc 4739 4739 264852
    (2005, 'Men', 'des'): 2005 Men des 4739 4739 264852
    (2006, 'Men', 'asc'): 2006 Men asc 4821 4821 270936
    (2006, 'Men', 'des'): 2006 Men des 4821 4821 270936
    (2007, 'Men', 'asc'): 2007 Men asc 5107 5107 302344
    (2007, 'Men', 'des'): 2007 Men des 5107 5107 302344
    (2008, 'Men', 'asc'): 2008 Men asc 5227 5227 311728
    (2008, 'Men', 'des'): 2008 Men des 5227 5227 311728
    (2009, 'Men', 'asc'): 2009 Men asc 5313 5313 317646
    (2009, 'Men', 'des'): 2009 Men des 5313 5313 317646
    (2010, 'Men', 'asc'): 2010 Men asc 5327 5327 318810
    (2010, 'Men', 'des'): 2010 Men des 5327 5327 318810
    (2010, 'Women', 'asc'): 2010 Women asc 5100 5100 295416
    (2010, 'Women', 'des'): 2010 Women des 5100 5100 295416
    (2011, 'Men', 'asc'): 2011 Men asc 5313 5313 319002
    (2011, 'Men', 'des'): 2011 Men des 5313 5313 319002
    (2011, 'Women', 'asc'): 2011 Women asc 5147 5147 300570
    (2011, 'Women', 'des'): 2011 Women des 5147 5147 300570
    (2012, 'Men', 'asc'): 2012 Men asc 5320 5320 319980
    (2012, 'Men', 'des'): 2012 Men des 5320 5320 319980
    (2012, 'Women', 'asc'): 2012 Women asc 5113 5113 297430
    (2012, 'Women', 'des'): 2012 Women des 5113 5113 297430
    (2013, 'Men', 'asc'): 2013 Men asc 5387 5387 325874
    (2013, 'Men', 'des'): 2013 Men des 5387 5387 325874
    (2013, 'Women', 'asc'): 2013 Women asc 5247 5247 310242
    (2013, 'Women', 'des'): 2013 Women des 5247 5247 310242
    (2014, 'Men', 'asc'): 2014 Men asc 5429 5429 327584
    (2014, 'Men', 'des'): 2014 Men des 5429 5429 327584
    (2014, 'Women', 'asc'): 2014 Women asc 5315 5315 314988
    (2014, 'Women', 'des'): 2014 Women des 5315 5315 314988
    (2015, 'Men', 'asc'): 2015 Men asc 5421 5421 326338
    (2015, 'Men', 'des'): 2015 Men des 5421 5421 326338
    (2015, 'Women', 'asc'): 2015 Women asc 5277 5277 310402
    (2015, 'Women', 'des'): 2015 Women des 5277 5277 310402
    (2016, 'Men', 'asc'): 2016 Men asc 5436 5436 327952
    (2016, 'Men', 'des'): 2016 Men des 5436 5436 327952
    (2016, 'Women', 'asc'): 2016 Women asc 5272 5272 309708
    (2016, 'Women', 'des'): 2016 Women des 5272 5272 309708
    (2017, 'Men', 'asc'): 2017 Men asc 5462 5462 331238
    (2017, 'Men', 'des'): 2017 Men des 5462 5462 331238
    (2017, 'Women', 'asc'): 2017 Women asc 5273 5273 309948
    (2017, 'Women', 'des'): 2017 Women des 5273 5273 309948
    (2018, 'Men', 'asc'): 2018 Men asc 5472 5472 332390
    (2018, 'Men', 'des'): 2018 Men des 5472 5472 332390
    (2018, 'Women', 'asc'): 2018 Women asc 5272 5272 309782
    (2018, 'Women', 'des'): 2018 Women des 5272 5272 309782
    (2019, 'Men', 'asc'): 2019 Men asc 5530 5530 337450
    (2019, 'Men', 'des'): 2019 Men des 5530 5530 337450
    (2019, 'Women', 'asc'): 2019 Women asc 5303 5303 311816
    (2019, 'Women', 'des'): 2019 Women des 5303 5303 311816
    (2020, 'Men', 'asc'): 2020 Men asc 5328 5328 311926
    (2020, 'Men', 'des'): 2020 Men des 5328 5328 311926
    (2020, 'Women', 'asc'): 2020 Women asc 5171 5171 295280
    (2020, 'Women', 'des'): 2020 Women des 5171 5171 295280
    (2021, 'Men', 'asc'): 2021 Men asc 3921 3921 176238
    (2021, 'Men', 'des'): 2021 Men des 3921 3921 176238
    (2021, 'Women', 'asc'): 2021 Women asc 3619 3619 153376
    (2021, 'Women', 'des'): 2021 Women des 3619 3619 153376
    (2022, 'Men', 'asc'): 2022 Men asc 5412 5412 319218
    (2022, 'Men', 'des'): 2022 Men des 5412 5412 319218
    (2022, 'Women', 'asc'): 2022 Women asc 5127 5127 287716
    (2022, 'Women', 'des'): 2022 Women des 5127 5127 287716
    (2023, 'Men', 'asc'): 2023 Men asc 5669 5669 344944
    (2023, 'Men', 'des'): 2023 Men des 5669 5669 344944
    (2023, 'Women', 'asc'): 2023 Women asc 5441 5441 319184
    (2023, 'Women', 'des'): 2023 Women des 5441 5441 319184
    (2024, 'Men', 'asc'): 2024 Men asc 5674 5674 346466
    (2024, 'Men', 'des'): 2024 Men des 5674 5674 346466
    (2024, 'Women', 'asc'): 2024 Women asc 5481 5481 324772
    (2024, 'Women', 'des'): 2024 Women des 5481 5481 324772
    (2025, 'Men', 'asc'): 2025 Men asc 5641 5641 339596
    (2025, 'Men', 'des'): 2025 Men des 5641 5641 339596
    (2025, 'Women', 'asc'): 2025 Women asc 5444 5444 317808
    (2025, 'Women', 'des'): 2025 Women des 5444 5444 317808



```python
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
```

    2003 Men asc 4680 4680 260196
    
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama    Oklahoma          6  1.0          0.0     0.0        22.0    10.0    0.0      68.0    27.0    58.0      3.0     14.0    11.0    18.0   14.0   24.0    13.0   23.0     7.0     1.0   22.0      62.0    22.0    53.0      2.0     10.0    16.0    22.0   10.0   22.0     8.0   18.0     9.0     2.0   20.0
    1         1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis    Syracuse          7  1.0          0.0     0.0        22.0    10.0    0.0      70.0    26.0    62.0      8.0     20.0    10.0    19.0   15.0   28.0    16.0   13.0     4.0     4.0   18.0      63.0    24.0    67.0      6.0     24.0     9.0    20.0   20.0   25.0     7.0   12.0     8.0     6.0   16.0
    2         2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington   Wisconsin        -26  1.0          0.0    -1.0        22.0    11.0    0.0      55.0    20.0    46.0      3.0     11.0    12.0    17.0    6.0   22.0     8.0   19.0     4.0     3.0   25.0      81.0    26.0    57.0      6.0     12.0    23.0    27.0   12.0   24.0    12.0    9.0     9.0     3.0   18.0
    3         3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia       Texas         -6  1.0          0.0     0.0        22.0    11.0    0.0      71.0    24.0    62.0      6.0     16.0    17.0    27.0   21.0   15.0    12.0   10.0     7.0     1.0   14.0      77.0    30.0    61.0      6.0     14.0    11.0    13.0   17.0   22.0    12.0   14.0     4.0     4.0   20.0
    4         4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette   Villanova         12  1.0          0.0     0.0        22.0    11.0    0.0      73.0    24.0    58.0      8.0     18.0    17.0    29.0   17.0   26.0    15.0   10.0     5.0     2.0   25.0      61.0    22.0    73.0      3.0     26.0    14.0    23.0   31.0   22.0     9.0   12.0     2.0     5.0   23.0
    ...     ...                 ...     ...        ...        ...        ...           ...         ...        ...  ...          ...     ...         ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1277       1400   Michigan St       Texas         -9  1.0          1.0     0.0        22.0   146.0    0.0      76.0    25.0    54.0      8.0     16.0    18.0    26.0   12.0   26.0    11.0   12.0     1.0     2.0   28.0      85.0    25.0    51.0      6.0     14.0    29.0    38.0    8.0   20.0    16.0    4.0     6.0     4.0   22.0
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1328       1393      Oklahoma    Syracuse        -16  1.0          1.0     0.0        22.0   146.0    0.0      47.0    18.0    58.0      5.0     28.0     6.0    11.0   14.0   14.0    10.0   19.0    13.0     4.0   16.0      63.0    25.0    48.0      2.0     12.0    11.0    19.0   12.0   28.0    16.0   24.0    13.0     2.0   13.0
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1242       1266        Kansas   Marquette         33  1.0          1.0     0.0        22.0   152.0    0.0      94.0    38.0    71.0      8.0     19.0    10.0    17.0   19.0   33.0    22.0   12.0     8.0     5.0   15.0      61.0    23.0    74.0      3.0     16.0    12.0    18.0   21.0   18.0     7.0   11.0     7.0     3.0   17.0
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1393       1400      Syracuse       Texas         11  1.0          1.0     0.0        22.0   152.0    0.0      95.0    32.0    56.0      7.0     13.0    24.0    31.0    9.0   28.0    14.0   13.0     9.0     2.0   21.0      84.0    27.0    63.0     10.0     21.0    20.0    32.0   17.0   17.0    20.0   14.0     6.0     2.0   26.0
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1242       1393        Kansas    Syracuse         -3  1.0          1.0     0.0        22.0   154.0    0.0      78.0    31.0    71.0      4.0     20.0    12.0    30.0   26.0   26.0    18.0   18.0     9.0     4.0   16.0      81.0    30.0    63.0     11.0     18.0    10.0    17.0   11.0   25.0    13.0   17.0    10.0     7.0   22.0
    
    [4680 rows x 43 columns]
    
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT   Le_Score     Le_FGM     Le_FGA   Le_FGM3    Le_FGA3     Le_FTM     Le_FTA      Le_OR      Le_DR     Le_Ast      Le_TO    Le_Stl    Le_Blk      Le_PF   Ri_Score     Ri_FGM     Ri_FGA   Ri_FGM3    Ri_FGA3     Ri_FTM     Ri_FTA      Ri_OR      Ri_DR     Ri_Ast      Ri_TO    Ri_Stl    Ri_Blk      Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama    Oklahoma          6  1.0          0.0     0.0        22.0    10.0    0.0  69.071426  23.857143  56.964287  6.428571  19.785715  14.928572  20.857143  13.392858  23.785715  12.107142  12.750000  6.428571  3.964286  18.000000  70.606064  25.181818  56.333332  7.515152  19.030304  12.727273  18.393940  12.454545  24.484848  14.363636  11.939394  6.909091  3.666667  18.151516
    1         1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis    Syracuse          7  1.0          0.0     0.0        22.0    10.0    0.0  74.551727  26.241379  60.241379  6.965517  20.344828  15.103448  22.965517  14.241380  25.758621  16.448277  13.758620  7.310345  5.103448  18.758621  80.058823  29.294117  61.147060  5.294117  15.176471  16.176470  23.117647  13.470589  27.000000  15.176471  14.176471  8.529411  7.088235  16.882353
    2         2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington   Wisconsin        -26  1.0          0.0    -1.0        22.0    11.0    0.0  70.500000  24.035715  51.392857  4.750000  13.428572  17.678572  24.071428   9.500000  22.892857  15.035714  16.964285  7.821429  2.714286  21.357143  69.967743  24.903225  53.741936  6.290323  17.774193  13.870968  19.096775  10.129032  22.419355  13.258064  10.483871  6.838710  2.741935  14.806452
    3         3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia       Texas         -6  1.0          0.0     0.0        22.0    11.0    0.0  79.500000  28.692308  61.423077  6.730769  17.692308  15.384615  21.423077  12.500000  24.961538  18.153847  11.615385  7.653846  4.500000  17.307692  79.406250  27.687500  62.312500  5.937500  16.750000  18.093750  25.156250  16.062500  26.031250  14.562500  13.031250  6.312500  3.656250  20.500000
    4         4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette   Villanova         12  1.0          0.0     0.0        22.0    11.0    0.0  78.625000  27.406250  56.656250  6.062500  15.312500  17.750000  22.812500  12.812500  23.593750  16.187500  13.375000  6.031250  3.593750  18.656250  72.586205  24.931034  58.586208  6.793103  18.862068  15.931034  22.241379  14.137931  23.758621  13.206897  16.172413  7.689655  3.344828  20.827587
    ...     ...                 ...     ...        ...        ...        ...           ...         ...        ...  ...          ...     ...         ...     ...    ...        ...        ...        ...       ...        ...        ...        ...        ...        ...        ...        ...       ...       ...        ...        ...        ...        ...       ...        ...        ...        ...        ...        ...        ...        ...       ...       ...        ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1277       1400   Michigan St       Texas         -9  1.0          1.0     0.0        22.0   146.0    0.0  67.382355  23.147058  51.205883  5.029412  13.558824  16.058823  21.794117  10.617647  24.323530  13.617647  14.411765  6.352941  3.500000  19.911764  79.156250  27.843750  62.625000  5.937500  16.750000  17.531250  24.375000  16.343750  26.093750  14.437500  13.343750  6.250000  3.656250  20.437500
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1328       1393      Oklahoma    Syracuse        -16  1.0          1.0     0.0        22.0   146.0    0.0  71.060608  25.303030  56.181820  7.424242  18.484848  13.030303  18.727272  12.333333  24.727272  14.303030  11.909091  6.787879  3.606061  18.272728  80.058823  29.264706  61.705883  5.411765  15.529411  16.117647  23.147058  13.705882  26.911764  14.911765  13.823529  8.382353  7.205883  16.970589
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1242       1266        Kansas   Marquette         33  1.0          1.0     0.0        22.0   152.0    0.0  81.057144  30.400000  62.400002  4.685714  14.200000  15.571428  23.828571  14.485714  26.628571  16.685715  14.885715  9.828571  4.885714  16.857143  79.000000  27.437500  56.156250  6.218750  15.375000  17.906250  23.156250  12.687500  23.843750  16.437500  13.343750  5.968750  3.562500  18.906250
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1393       1400      Syracuse       Texas         11  1.0          1.0     0.0        22.0   152.0    0.0  79.117645  29.058823  61.470589  5.264706  15.500000  15.735294  22.794117  13.794118  26.911764  14.970589  14.147058  8.500000  7.205883  16.735294  79.187500  27.781250  62.250000  5.812500  16.531250  17.812500  24.562500  16.062500  26.187500  14.312500  13.031250  6.250000  3.718750  20.312500
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1242       1393        Kansas    Syracuse         -3  1.0          1.0     0.0        22.0   154.0    0.0  81.514282  30.600000  62.400002  4.800000  14.171429  15.514286  23.457144  14.285714  26.828571  16.799999  14.714286  9.800000  4.914286  16.828571  79.529411  29.117647  61.264706  5.147059  15.352942  16.147058  23.205883  13.735294  27.000000  15.000000  14.029411  8.470589  7.058824  16.705883
    
    [4680 rows x 43 columns]
    
    
    
    2003 Men des 4680 4680 260196
    
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID Le_TeamName   Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1328       1104    Oklahoma       Alabama         -6  1.0          0.0     0.0        22.0    10.0    0.0      62.0    22.0    53.0      2.0     10.0    16.0    22.0   10.0   22.0     8.0   18.0     9.0     2.0   20.0      68.0    27.0    58.0      3.0     14.0    11.0    18.0   14.0   24.0    13.0   23.0     7.0     1.0   22.0
    1         1  2003_010_1272_1393    2003 2002-11-14       1393       1272    Syracuse       Memphis         -7  1.0          0.0     0.0        22.0    10.0    0.0      63.0    24.0    67.0      6.0     24.0     9.0    20.0   20.0   25.0     7.0   12.0     8.0     6.0   16.0      70.0    26.0    62.0      8.0     20.0    10.0    19.0   15.0   28.0    16.0   13.0     4.0     4.0   18.0
    2         2  2003_011_1186_1458    2003 2002-11-15       1458       1186   Wisconsin  E Washington         26  1.0          0.0     1.0        22.0    11.0    0.0      81.0    26.0    57.0      6.0     12.0    23.0    27.0   12.0   24.0    12.0    9.0     9.0     3.0   18.0      55.0    20.0    46.0      3.0     11.0    12.0    17.0    6.0   22.0     8.0   19.0     4.0     3.0   25.0
    3         3  2003_011_1208_1400    2003 2002-11-15       1400       1208       Texas       Georgia          6  1.0          0.0     0.0        22.0    11.0    0.0      77.0    30.0    61.0      6.0     14.0    11.0    13.0   17.0   22.0    12.0   14.0     4.0     4.0   20.0      71.0    24.0    62.0      6.0     16.0    17.0    27.0   21.0   15.0    12.0   10.0     7.0     1.0   14.0
    4         4  2003_011_1266_1437    2003 2002-11-15       1437       1266   Villanova     Marquette        -12  1.0          0.0     0.0        22.0    11.0    0.0      61.0    22.0    73.0      3.0     26.0    14.0    23.0   31.0   22.0     9.0   12.0     2.0     5.0   23.0      73.0    24.0    58.0      8.0     18.0    17.0    29.0   17.0   26.0    15.0   10.0     5.0     2.0   25.0
    ...     ...                 ...     ...        ...        ...        ...         ...           ...        ...  ...          ...     ...         ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1400       1277       Texas   Michigan St          9  1.0          1.0     0.0        22.0   146.0    0.0      85.0    25.0    51.0      6.0     14.0    29.0    38.0    8.0   20.0    16.0    4.0     6.0     4.0   22.0      76.0    25.0    54.0      8.0     16.0    18.0    26.0   12.0   26.0    11.0   12.0     1.0     2.0   28.0
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1393       1328    Syracuse      Oklahoma         16  1.0          1.0     0.0        22.0   146.0    0.0      63.0    25.0    48.0      2.0     12.0    11.0    19.0   12.0   28.0    16.0   24.0    13.0     2.0   13.0      47.0    18.0    58.0      5.0     28.0     6.0    11.0   14.0   14.0    10.0   19.0    13.0     4.0   16.0
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1266       1242   Marquette        Kansas        -33  1.0          1.0     0.0        22.0   152.0    0.0      61.0    23.0    74.0      3.0     16.0    12.0    18.0   21.0   18.0     7.0   11.0     7.0     3.0   17.0      94.0    38.0    71.0      8.0     19.0    10.0    17.0   19.0   33.0    22.0   12.0     8.0     5.0   15.0
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1400       1393       Texas      Syracuse        -11  1.0          1.0     0.0        22.0   152.0    0.0      84.0    27.0    63.0     10.0     21.0    20.0    32.0   17.0   17.0    20.0   14.0     6.0     2.0   26.0      95.0    32.0    56.0      7.0     13.0    24.0    31.0    9.0   28.0    14.0   13.0     9.0     2.0   21.0
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1393       1242    Syracuse        Kansas          3  1.0          1.0     0.0        22.0   154.0    0.0      81.0    30.0    63.0     11.0     18.0    10.0    17.0   11.0   25.0    13.0   17.0    10.0     7.0   22.0      78.0    31.0    71.0      4.0     20.0    12.0    30.0   26.0   26.0    18.0   18.0     9.0     4.0   16.0
    
    [4680 rows x 43 columns]
    
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID Le_TeamName   Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT   Le_Score     Le_FGM     Le_FGA   Le_FGM3    Le_FGA3     Le_FTM     Le_FTA      Le_OR      Le_DR     Le_Ast      Le_TO    Le_Stl    Le_Blk      Le_PF   Ri_Score     Ri_FGM     Ri_FGA   Ri_FGM3    Ri_FGA3     Ri_FTM     Ri_FTA      Ri_OR      Ri_DR     Ri_Ast      Ri_TO    Ri_Stl    Ri_Blk      Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1328       1104    Oklahoma       Alabama         -6  1.0          0.0     0.0        22.0    10.0    0.0  70.606064  25.181818  56.333332  7.515152  19.030304  12.727273  18.393940  12.454545  24.484848  14.363636  11.939394  6.909091  3.666667  18.151516  69.071426  23.857143  56.964287  6.428571  19.785715  14.928572  20.857143  13.392858  23.785715  12.107142  12.750000  6.428571  3.964286  18.000000
    1         1  2003_010_1272_1393    2003 2002-11-14       1393       1272    Syracuse       Memphis         -7  1.0          0.0     0.0        22.0    10.0    0.0  80.058823  29.294117  61.147060  5.294117  15.176471  16.176470  23.117647  13.470589  27.000000  15.176471  14.176471  8.529411  7.088235  16.882353  74.551727  26.241379  60.241379  6.965517  20.344828  15.103448  22.965517  14.241380  25.758621  16.448277  13.758620  7.310345  5.103448  18.758621
    2         2  2003_011_1186_1458    2003 2002-11-15       1458       1186   Wisconsin  E Washington         26  1.0          0.0     1.0        22.0    11.0    0.0  69.967743  24.903225  53.741936  6.290323  17.774193  13.870968  19.096775  10.129032  22.419355  13.258064  10.483871  6.838710  2.741935  14.806452  70.500000  24.035715  51.392857  4.750000  13.428572  17.678572  24.071428   9.500000  22.892857  15.035714  16.964285  7.821429  2.714286  21.357143
    3         3  2003_011_1208_1400    2003 2002-11-15       1400       1208       Texas       Georgia          6  1.0          0.0     0.0        22.0    11.0    0.0  79.406250  27.687500  62.312500  5.937500  16.750000  18.093750  25.156250  16.062500  26.031250  14.562500  13.031250  6.312500  3.656250  20.500000  79.500000  28.692308  61.423077  6.730769  17.692308  15.384615  21.423077  12.500000  24.961538  18.153847  11.615385  7.653846  4.500000  17.307692
    4         4  2003_011_1266_1437    2003 2002-11-15       1437       1266   Villanova     Marquette        -12  1.0          0.0     0.0        22.0    11.0    0.0  72.586205  24.931034  58.586208  6.793103  18.862068  15.931034  22.241379  14.137931  23.758621  13.206897  16.172413  7.689655  3.344828  20.827587  78.625000  27.406250  56.656250  6.062500  15.312500  17.750000  22.812500  12.812500  23.593750  16.187500  13.375000  6.031250  3.593750  18.656250
    ...     ...                 ...     ...        ...        ...        ...         ...           ...        ...  ...          ...     ...         ...     ...    ...        ...        ...        ...       ...        ...        ...        ...        ...        ...        ...        ...       ...       ...        ...        ...        ...        ...       ...        ...        ...        ...        ...        ...        ...        ...       ...       ...        ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1400       1277       Texas   Michigan St          9  1.0          1.0     0.0        22.0   146.0    0.0  79.156250  27.843750  62.625000  5.937500  16.750000  17.531250  24.375000  16.343750  26.093750  14.437500  13.343750  6.250000  3.656250  20.437500  67.382355  23.147058  51.205883  5.029412  13.558824  16.058823  21.794117  10.617647  24.323530  13.617647  14.411765  6.352941  3.500000  19.911764
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1393       1328    Syracuse      Oklahoma         16  1.0          1.0     0.0        22.0   146.0    0.0  80.058823  29.264706  61.705883  5.411765  15.529411  16.117647  23.147058  13.705882  26.911764  14.911765  13.823529  8.382353  7.205883  16.970589  71.060608  25.303030  56.181820  7.424242  18.484848  13.030303  18.727272  12.333333  24.727272  14.303030  11.909091  6.787879  3.606061  18.272728
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1266       1242   Marquette        Kansas        -33  1.0          1.0     0.0        22.0   152.0    0.0  79.000000  27.437500  56.156250  6.218750  15.375000  17.906250  23.156250  12.687500  23.843750  16.437500  13.343750  5.968750  3.562500  18.906250  81.057144  30.400000  62.400002  4.685714  14.200000  15.571428  23.828571  14.485714  26.628571  16.685715  14.885715  9.828571  4.885714  16.857143
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1400       1393       Texas      Syracuse        -11  1.0          1.0     0.0        22.0   152.0    0.0  79.187500  27.781250  62.250000  5.812500  16.531250  17.812500  24.562500  16.062500  26.187500  14.312500  13.031250  6.250000  3.718750  20.312500  79.117645  29.058823  61.470589  5.264706  15.500000  15.735294  22.794117  13.794118  26.911764  14.970589  14.147058  8.500000  7.205883  16.735294
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1393       1242    Syracuse        Kansas          3  1.0          1.0     0.0        22.0   154.0    0.0  79.529411  29.117647  61.264706  5.147059  15.352942  16.147058  23.205883  13.735294  27.000000  15.000000  14.029411  8.470589  7.058824  16.705883  81.514282  30.600000  62.400002  4.800000  14.171429  15.514286  23.457144  14.285714  26.828571  16.799999  14.714286  9.800000  4.914286  16.828571
    
    [4680 rows x 43 columns]



```python
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
```


```python
nodes = pd.concat([graph.nodes for graph in graphs.values()])
scaler_x.fit(nodes.loc[:, "SeasonsAgo":])
scaler_y.fit(nodes[["Le_Margin"]])

for graph in graphs.values():
  graph.nodes = scale_values(graph.nodes)
  graph.sea = scale_values(graph.sea)

print(f"{graphs[(2003, 'Men', 'asc')].nodes}")
print(f"{graphs[(2003, 'Men', 'asc')].sea}")
```

          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName Ri_TeamName  Le_Margin      Le_y  Men  NCAATourney  Le_Loc  SeasonsAgo    DayNum     NumOT  Le_Score    Le_FGM    Le_FGA   Le_FGM3   Le_FGA3    Le_FTM    Le_FTA     Le_OR     Le_DR    Le_Ast     Le_TO    Le_Stl    Le_Blk     Le_PF  Ri_Score    Ri_FGM    Ri_FGA   Ri_FGM3   Ri_FGA3    Ri_FTM    Ri_FTA     Ri_OR     Ri_DR    Ri_Ast     Ri_TO    Ri_Stl    Ri_Blk     Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama    Oklahoma          6  0.364316  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.021993  0.566312  0.102891 -1.084275 -0.794976 -0.364918 -0.108382  0.691001  0.002141 -0.003511  1.793540 -0.013958 -1.005587  0.935628 -0.435726 -0.414578 -0.539242 -1.411433 -1.427122  0.475545  0.406779 -0.208418 -0.373919 -1.106682  0.755983  0.592462 -0.567787  0.494275
    1         1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis    Syracuse          7  0.425036  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.174566  0.370134  0.616597  0.551516  0.153242 -0.533011  0.020409  0.915856  0.754260  0.658391 -0.281574 -0.923589  0.307814  0.052923 -0.359440 -0.022222  1.258731 -0.102800  0.785388 -0.701103  0.149199  2.040129  0.190171 -1.327316 -0.489085  0.289252  1.183415 -0.388430
    2         2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington   Wisconsin        -26 -1.578703  1.0          0.0    -1.0    2.069352 -1.642327 -0.214915 -0.969732 -0.806934 -1.438229 -1.084275 -1.269085 -0.196825 -0.237172 -1.107837 -0.373919 -1.106682  0.963494 -0.923589 -0.129986  1.597657  1.013718  0.370134 -0.025536 -0.102800 -1.111049  1.652194  1.050730  0.241292  0.002141 -0.224145 -1.111619  0.592462 -0.129986  0.052923
    3         3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia       Texas         -6 -0.364316  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.250853 -0.022222  0.616597 -0.102800 -0.478903  0.643638  1.050730  2.264984 -1.690127 -0.224145 -0.904108 -0.013958 -1.005587 -0.829782  0.708572  1.154847  0.488171 -0.102800 -0.794976 -0.364918 -0.752333  1.365565 -0.373919 -0.224145 -0.074062 -0.923589  0.307814  0.494275
    4         4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette   Villanova         12  0.728632  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.403426 -0.022222  0.102891  0.551516 -0.162830  0.643638  1.308311  1.365565  0.378201  0.437757 -0.904108 -0.620379 -0.567787  1.597657 -0.512013 -0.414578  2.029291 -1.084275  1.101461  0.139360  0.535570  4.513531 -0.373919 -0.886047 -0.489085 -1.530009  0.745615  1.156304
    ...     ...                 ...     ...        ...        ...        ...           ...         ...        ...       ...  ...          ...     ...         ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1277       1400   Michigan St       Texas         -9 -0.546474  1.0          1.0     0.0    2.069352  2.043746 -0.214915  0.632286  0.173956 -0.410816  0.551516 -0.478903  0.811731  0.921940  0.241292  0.378201 -0.444779 -0.489085 -1.833219 -0.567787  2.259686  1.318865  0.173956 -0.796096 -0.102800 -0.794976  2.660750  2.467423 -0.658127 -0.749978  0.658391 -2.149176 -0.317169  0.307814  0.935628
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1328       1393      Oklahoma    Syracuse        -16 -0.971510  1.0          1.0     0.0    2.069352  2.043746 -0.214915 -1.580025 -1.199291  0.102891 -0.429959  1.417534 -1.205381 -1.009913  0.691001 -1.878157 -0.665413  0.963494  1.805302  0.307814 -0.388430 -0.359440  0.173956 -1.181376 -1.411433 -1.111049 -0.364918  0.020409  0.241292  0.754260  0.658391  2.001051  1.805302 -0.567787 -1.050459
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1242       1266        Kansas   Marquette         33  2.003739  1.0          1.0     0.0    2.069352  2.207571 -0.214915  2.005444  2.724271  1.772437  0.551516 -0.004794 -0.533011 -0.237172  1.815275  1.694410  1.982195 -0.489085  0.289252  0.745615 -0.609106 -0.512013 -0.218400  2.157717 -1.084275 -0.478903 -0.196825 -0.108382  2.264984 -1.126038 -1.327316 -0.696597 -0.013958 -0.129986 -0.167754
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1393       1400      Syracuse       Texas         11  0.667913  1.0          1.0     0.0    2.069352  2.207571 -0.214915  2.081730  1.547203 -0.153962  0.224358 -0.953012  1.820287  1.565891 -0.433273  0.754260  0.217123 -0.281574  0.592462 -0.567787  0.714952  1.242578  0.566312  0.745024  1.205832  0.311279  1.147916  1.694682  1.365565 -1.314068  1.540927 -0.074062 -0.317169 -0.567787  1.818333
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1242       1393        Kansas    Syracuse         -3 -0.182158  1.0          1.0     0.0    2.069352  2.262179 -0.214915  0.784859  1.351025  1.772437 -0.757117  0.153242 -0.196825  1.437101  3.389258  0.378201  1.099659  0.755983  0.592462  0.307814 -0.388430  1.013718  1.154847  0.745024  1.532990 -0.162830 -0.533011 -0.237172  0.016437  0.190171 -0.003511  0.548472  0.895672  1.621215  0.935628
    
    [4680 rows x 44 columns]
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName Ri_TeamName  Le_Margin      Le_y  Men  NCAATourney  Le_Loc  SeasonsAgo    DayNum     NumOT  Le_Score    Le_FGM    Le_FGA   Le_FGM3   Le_FGA3    Le_FTM    Le_FTA     Le_OR     Le_DR    Le_Ast     Le_TO    Le_Stl    Le_Blk     Le_PF  Ri_Score    Ri_FGM    Ri_FGA   Ri_FGM3   Ri_FGA3    Ri_FTM    Ri_FTA     Ri_OR     Ri_DR    Ri_Ast     Ri_TO    Ri_Stl    Ri_Blk     Ri_PF
    0         0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama    Oklahoma          6  0.364316  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.103729 -0.050247 -0.030122  0.037410  0.119378  0.295446  0.259591  0.554482 -0.038151 -0.200506 -0.333452 -0.187221  0.292178  0.052923  0.220801  0.209625 -0.111154  0.392894 -0.000005 -0.074576 -0.057646  0.343498  0.093307  0.297353 -0.501662 -0.041523  0.161881  0.086359
    1         1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis    Syracuse          7  0.425036  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.521802  0.417487  0.390744  0.213076  0.207738  0.324842  0.531128  0.745276  0.332814  0.757296 -0.124152  0.080141  0.790904  0.220333  0.941919  1.016368  0.507057 -0.333736 -0.609051  0.505209  0.550721  0.571960  0.566231  0.476692 -0.037443  0.449775  1.659845 -0.193715
    2         2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington   Wisconsin        -26 -1.578703  1.0          0.0    -1.0    2.069352 -1.642327 -0.214915  0.212710 -0.015215 -0.745642 -0.511748 -0.885283  0.757701  0.673559 -0.320845 -0.206035  0.445637  0.541060  0.235107 -0.255072  0.793765  0.172105  0.154971 -0.443958 -0.007819 -0.198516  0.117671  0.032872 -0.179404 -0.295067  0.053427 -0.803699 -0.062863 -0.242967 -0.651818
    3         3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia       Texas         -6 -0.364316  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.899289  0.898306  0.542505  0.136277 -0.211457  0.372104  0.332477  0.353719  0.182939  1.133603 -0.568897  0.184294  0.526714 -0.099853  0.892137  0.701185  0.656731 -0.123248 -0.360376  0.827489  0.813273  1.154764  0.384077  0.341229 -0.275089 -0.222415  0.157320  0.604614
    4         4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette   Villanova         12  0.728632  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.832538  0.646010 -0.069682 -0.082353 -0.587553  0.769708  0.511421  0.423986 -0.074246  0.699760 -0.203757 -0.307693  0.129958  0.197742  0.371859  0.160427  0.178176  0.156670 -0.026592  0.463953  0.437867  0.722015 -0.043245  0.042137  0.376738  0.195152  0.020979  0.676904
    ...     ...                 ...     ...        ...        ...        ...           ...         ...        ...       ...  ...          ...     ...         ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    4675   4675  2003_146_1277_1400    2003 2003-03-30       1277       1400   Michigan St       Texas         -9 -0.546474  1.0          1.0     0.0    2.069352  2.043746 -0.214915 -0.025125 -0.189550 -0.769655 -0.420336 -0.864698  0.485433  0.380264 -0.069537  0.062974  0.132763  0.011383 -0.210153  0.088914  0.474804  0.873065  0.731838  0.696864 -0.123248 -0.360376  0.732937  0.712656  1.218004  0.395829  0.313650 -0.210242 -0.241366  0.157320  0.590821
    4676   4676  2003_146_1328_1393    2003 2003-03-30       1328       1393      Oklahoma    Syracuse        -16 -0.971510  1.0          1.0     0.0    2.069352  2.043746 -0.214915  0.255476  0.233404 -0.130612  0.363152 -0.086207 -0.023639 -0.014716  0.316243  0.138890  0.283982 -0.507950 -0.078276  0.135347  0.113107  0.941919  1.010598  0.578825 -0.295246 -0.553273  0.495321  0.554509  0.624867  0.549640  0.418289 -0.110682  0.405185  1.711351 -0.174244
    4677   4677  2003_152_1242_1266    2003 2003-04-05       1242       1266        Kansas   Marquette         33  2.003739  1.0          1.0     0.0    2.069352  2.207571 -0.214915  1.018078  1.233318  0.667968 -0.532780 -0.763369  0.403506  0.642281  0.800216  0.496391  0.809683  0.109733  0.843693  0.695580 -0.199279  0.861145  0.652140 -0.133896 -0.031235 -0.577676  0.795972  0.555693  0.395879 -0.027239  0.754918 -0.210242 -0.326644  0.116276  0.252911
    4678   4678  2003_152_1393_1400    2003 2003-04-05       1393       1400      Syracuse       Texas         11  0.667913  1.0          1.0     0.0    2.069352  2.207571 -0.214915  0.870120  0.970208  0.548607 -0.343358 -0.557921  0.431050  0.509054  0.644708  0.549640  0.431268 -0.043546  0.440857  1.711351 -0.226168  0.875449  0.719576  0.648704 -0.164143 -0.394946  0.780213  0.736804  1.154764  0.413456  0.286071 -0.275089 -0.241366  0.184683  0.563237
    4679   4679  2003_154_1242_1393    2003 2003-04-07       1242       1393        Kansas    Syracuse         -3 -0.182158  1.0          1.0     0.0    2.069352  2.262179 -0.214915  1.052951  1.272554  0.667968 -0.495390 -0.767884  0.393900  0.594445  0.755245  0.533997  0.834898  0.074160  0.835030  0.708089 -0.205584  0.901532  0.981748  0.522166 -0.381847 -0.581162  0.500265  0.562085  0.631481  0.566231  0.437757 -0.067959  0.431939  1.646968 -0.232658
    
    [4680 rows x 44 columns]



```python
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
```


```python
print(f"{graphs[(2021, 'Men', 'asc')].nodes}")
print(f"{graphs[(2021, 'Men', 'asc')].train[0]}")
print()
print()
print()
print(f"{graphs[(2021, 'Men', 'asc')].edges}")
print(f"{graphs[(2021, 'Men', 'asc')].train[1]}")
```

          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID     Le_TeamName      Ri_TeamName  Le_Margin      Le_y  Men  NCAATourney  Le_Loc  SeasonsAgo    DayNum     NumOT  Le_Score    Le_FGM    Le_FGA   Le_FGM3   Le_FGA3    Le_FTM    Le_FTA     Le_OR     Le_DR    Le_Ast     Le_TO    Le_Stl    Le_Blk     Le_PF  Ri_Score    Ri_FGM    Ri_FGA   Ri_FGM3   Ri_FGA3    Ri_FTM    Ri_FTA     Ri_OR     Ri_DR    Ri_Ast     Ri_TO    Ri_Stl    Ri_Blk     Ri_PF
    0         0  2021_023_1101_1190    2021 2020-11-25       1101       1190     Abilene Chr             ETSU         23  1.396545  1.0          0.0     0.0   -0.894841 -1.314676 -0.214915  0.174566 -0.806934 -1.052949  2.187307  0.627352  0.643638  1.050730 -1.107837  0.002141  0.217123  0.133449  0.289252 -0.129986  0.714952 -1.580025 -1.787825 -0.796096 -1.084275  0.311279  0.139360 -0.237172  0.016437  0.566231 -1.768584  1.586029 -0.317169 -0.567787  1.818333
    1         1  2021_023_1104_1240    2021 2020-11-25       1104       1240         Alabama  Jacksonville St         24  1.457265  1.0          0.0     1.0   -0.894841 -1.314676 -0.214915  1.013718  1.154847  2.542997  0.224358  1.891643  0.139360  0.149199  1.815275  1.318350 -0.444779 -0.904108  0.289252 -0.129986 -0.167754 -0.817159 -1.003112  1.515584  0.878674  2.049679 -0.533011 -0.237172  1.590420  0.754260 -0.665413  1.171006 -0.620379 -0.567787  0.494275
    2         2  2021_023_1108_1412    2021 2020-11-25       1108       1412       Alcorn St              UAB        -49 -2.975249  1.0          0.0    -1.0   -0.894841 -1.314676 -0.214915 -1.351165 -1.003112  0.745024 -1.084275 -0.478903 -0.701103 -0.881123  0.466146 -1.126038 -2.430486  1.378517 -1.530009 -1.443388  1.156304  2.386877  1.351025  0.102891 -0.102800 -0.636940  2.996936  2.596213 -1.332691  0.566231 -0.444779 -0.696597  1.805302  2.934617 -0.388430
    3         3  2021_023_1111_1354    2021 2020-11-25       1111       1354  Appalachian St    S Carolina St         20  1.214387  1.0          0.0    -1.0   -0.894841 -1.314676 -0.214915  1.013718  1.547203  0.616597  0.224358  1.259497 -0.533011 -0.752333  0.466146  3.198648  0.437757  1.793540 -0.317169 -0.129986 -1.050459 -0.512013 -0.022222  2.928277  0.224358  2.049679 -1.205381 -1.009913  0.241292 -1.126038 -0.224145 -1.111619  1.805302 -1.443388 -0.829782
    4         4  2021_023_1113_1348    2021 2020-11-25       1113       1348      Arizona St     Rhode Island          6  0.364316  1.0          0.0     0.0   -0.894841 -1.314676 -0.214915  2.005444  0.566312  0.488171 -0.102800  0.153242  3.501214  3.497745  0.466146 -0.373919 -0.886047 -0.489085 -0.013958  0.307814  1.818333  1.547724  0.958669 -0.153962  0.878674 -0.162830  1.316009  2.081052 -0.882982  0.378201  0.217123  0.340960  0.289252 -0.567787  2.480362
    ...     ...                 ...     ...        ...        ...        ...             ...              ...        ...       ...  ...          ...     ...         ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    3916   3916  2021_148_1211_1425    2021 2021-03-30       1211       1425         Gonzaga              USC         19  1.153668  1.0          1.0     0.0   -0.894841  2.098354 -0.214915  1.318865  1.743381  1.130304  0.224358  0.311279 -0.196825 -0.237172  0.016437  0.566231  1.761561 -1.111619 -0.317169 -0.129986 -0.388430 -0.130580 -0.022222  0.616597 -0.757117 -0.636940  0.139360  0.020409 -0.882982 -0.749978 -0.886047 -1.111619 -0.013958 -1.443388 -1.050459
    3917   3917  2021_148_1276_1417    2021 2021-03-30       1276       1417        Michigan             UCLA         -2 -0.121439  1.0          1.0     0.0   -0.894841  2.098354 -0.214915 -1.427452 -0.806934 -0.796096 -1.084275 -1.269085 -1.205381 -1.009913 -0.658127  0.002141 -0.224145 -0.074062 -0.620379 -0.129986 -1.491811 -1.274878 -0.610756 -0.410816 -1.084275 -0.953012 -1.205381 -1.525074 -1.107837 -0.561948 -0.224145 -1.319131 -0.620379 -0.567787 -0.829782
    3918   3918  2021_152_1124_1222    2021 2021-04-03       1124       1222          Baylor          Houston         19  1.153668  1.0          1.0     0.0   -0.894841  2.207571 -0.214915  0.784859  0.958669 -0.282389  1.532990  0.785388 -0.701103 -0.752333  0.016437 -1.314068  2.202829 -1.319131 -0.317169 -1.443388  0.052923 -0.664586 -0.610756 -0.282389 -0.102800 -0.004794 -0.364918 -0.365962  0.466146 -2.254217 -0.665413 -0.904108 -0.923589  0.745615 -1.712488
    3919   3919  2021_152_1211_1417    2021 2021-04-03       1211       1417         Gonzaga             UCLA          3  0.182158  1.0          1.0     0.0   -0.894841  2.207571  3.270769  1.929157  2.528093  0.745024  0.224358  0.311279 -0.196825  0.149199 -1.557546 -0.938008  2.644097 -0.904108  0.289252 -0.129986 -0.388430  1.700297  1.939559  0.231318  0.551516 -0.320867  0.139360  0.277989 -0.882982  0.002141  1.761561 -1.111619 -0.923589 -1.005587 -0.388430
    3920   3920  2021_154_1124_1211    2021 2021-04-05       1124       1211          Baylor          Gonzaga         16  0.971510  1.0          1.0     0.0   -0.894841  2.262179 -0.214915  1.395151  1.154847  1.258731  1.205832  0.627352  0.475545 -0.108382  0.691001 -0.749978  1.099659 -1.526642  0.289252  0.745615  0.273599  0.174566  0.173956 -1.052949 -0.429959 -0.320867  0.307453  0.277989 -2.232110 -1.502098  0.658391 -0.074062 -0.923589 -0.129986 -0.167754
    
    [3921 rows x 44 columns]
          Index                 Key  Season       Date  Le_TeamID  Ri_TeamID     Le_TeamName      Ri_TeamName  Le_Margin      Le_y  Men  NCAATourney  Le_Loc  SeasonsAgo    DayNum     NumOT  Le_Score    Le_FGM    Le_FGA   Le_FGM3   Le_FGA3    Le_FTM    Le_FTA     Le_OR     Le_DR    Le_Ast     Le_TO    Le_Stl    Le_Blk     Le_PF  Ri_Score    Ri_FGM    Ri_FGA   Ri_FGM3   Ri_FGA3    Ri_FTM    Ri_FTA     Ri_OR     Ri_DR    Ri_Ast     Ri_TO    Ri_Stl    Ri_Blk     Ri_PF
    0         0  2021_023_1101_1190    2021 2020-11-25       1101       1190     Abilene Chr             ETSU         23  1.396545  1.0          0.0     0.0   -0.894841 -1.314676 -0.214915  0.174566 -0.806934 -1.052949  2.187307  0.627352  0.643638  1.050730 -1.107837  0.002141  0.217123  0.133449  0.289252 -0.129986  0.714952 -1.580025 -1.787825 -0.796096 -1.084275  0.311279  0.139360 -0.237172  0.016437  0.566231 -1.768584  1.586029 -0.317169 -0.567787  1.818333
    1         1  2021_023_1104_1240    2021 2020-11-25       1104       1240         Alabama  Jacksonville St         24  1.457265  1.0          0.0     1.0   -0.894841 -1.314676 -0.214915  1.013718  1.154847  2.542997  0.224358  1.891643  0.139360  0.149199  1.815275  1.318350 -0.444779 -0.904108  0.289252 -0.129986 -0.167754 -0.817159 -1.003112  1.515584  0.878674  2.049679 -0.533011 -0.237172  1.590420  0.754260 -0.665413  1.171006 -0.620379 -0.567787  0.494275
    2         2  2021_023_1108_1412    2021 2020-11-25       1108       1412       Alcorn St              UAB        -49 -2.975249  1.0          0.0    -1.0   -0.894841 -1.314676 -0.214915 -1.351165 -1.003112  0.745024 -1.084275 -0.478903 -0.701103 -0.881123  0.466146 -1.126038 -2.430486  1.378517 -1.530009 -1.443388  1.156304  2.386877  1.351025  0.102891 -0.102800 -0.636940  2.996936  2.596213 -1.332691  0.566231 -0.444779 -0.696597  1.805302  2.934617 -0.388430
    3         3  2021_023_1111_1354    2021 2020-11-25       1111       1354  Appalachian St    S Carolina St         20  1.214387  1.0          0.0    -1.0   -0.894841 -1.314676 -0.214915  1.013718  1.547203  0.616597  0.224358  1.259497 -0.533011 -0.752333  0.466146  3.198648  0.437757  1.793540 -0.317169 -0.129986 -1.050459 -0.512013 -0.022222  2.928277  0.224358  2.049679 -1.205381 -1.009913  0.241292 -1.126038 -0.224145 -1.111619  1.805302 -1.443388 -0.829782
    4         4  2021_023_1113_1348    2021 2020-11-25       1113       1348      Arizona St     Rhode Island          6  0.364316  1.0          0.0     0.0   -0.894841 -1.314676 -0.214915  2.005444  0.566312  0.488171 -0.102800  0.153242  3.501214  3.497745  0.466146 -0.373919 -0.886047 -0.489085 -0.013958  0.307814  1.818333  1.547724  0.958669 -0.153962  0.878674 -0.162830  1.316009  2.081052 -0.882982  0.378201  0.217123  0.340960  0.289252 -0.567787  2.480362
    ...     ...                 ...     ...        ...        ...        ...             ...              ...        ...       ...  ...          ...     ...         ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    3850   3850  2021_132_1104_1261    2021 2021-03-14       1104       1261         Alabama              LSU          1  0.060719  1.0          0.0     0.0   -0.894841  1.661486 -0.214915  0.937432  1.743381  2.414571  1.860149  2.681825 -1.877752 -1.782654  1.140710  1.318350  0.658391 -1.111619 -0.923589  2.059016 -0.609106  0.861145  1.154847  2.157717  1.205832  1.259497 -0.701103 -0.752333  0.915856  0.378201 -0.665413 -1.941665 -0.317169  0.745615 -1.050459
    3851   3851  2021_132_1153_1222    2021 2021-03-14       1153       1222      Cincinnati          Houston        -37 -2.246616  1.0          0.0     0.0   -0.894841  1.661486 -0.214915 -1.046019 -1.199291  1.001877  0.551516  2.207716 -0.533011 -0.494752  0.691001 -1.314068 -0.444779 -0.904108 -1.530009  0.745615 -1.271135  1.776584  2.528093  1.130304  1.532990  0.627352 -1.205381 -1.525074 -0.208418  0.378201  2.423463 -1.734154  0.289252  0.307814 -0.388430
    3852   3852  2021_132_1159_1259    2021 2021-03-14       1159       1259         Colgate        Loyola MD         13  0.789352  1.0          0.0     1.0   -0.894841  1.661486 -0.214915  1.318865  1.351025  0.231318  2.514465  0.627352 -0.701103 -0.623542 -1.107837 -0.749978  1.099659 -1.319131  0.592462 -1.005587 -0.609106  0.327139  0.958669  0.616597 -0.429959  0.469315 -0.701103 -0.365962  0.241292 -0.938008 -0.224145 -0.489085 -1.530009 -1.005587 -1.050459
    3853   3853  2021_132_1228_1326    2021 2021-03-14       1228       1326        Illinois          Ohio St          3  0.182158  1.0          0.0     0.0   -0.894841  1.661486  3.270769  1.776584  0.958669  0.873451  0.878674  0.311279  1.820287  1.694682  0.466146  0.566231  0.437757 -0.904108 -0.620379 -0.567787  1.156304  1.547724  0.958669  1.001877  0.551516  0.943424  1.484102  1.437101 -0.658127  0.566231 -0.444779 -1.319131 -0.013958  0.307814  1.818333
    3854   3854  2021_132_1382_1433    2021 2021-03-14       1382       1433  St Bonaventure              VCU          9  0.546474  1.0          0.0     0.0   -0.894841  1.661486 -0.214915  0.479712  0.173956  0.616597  0.551516 -0.004794  0.475545  0.149199  0.016437  0.378201 -0.224145 -1.111619 -1.226799  0.307814 -0.609106 -0.206867 -0.414578 -0.410816 -0.102800 -0.162830  0.307453  0.149199 -1.782401 -0.185889 -0.886047 -1.111619 -0.923589  0.307814  0.052923
    
    [3855 rows x 44 columns]
    
    
    
            SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                 0          2021                  0           82          2021                  0     2        1.0    1.0
    1                 0          2021                  0          104          2021                  0     0        1.0    2.0
    2                 0          2021                  0          123          2021                  0     3        1.0    2.0
    3                 0          2021                  0          155          2021                  0     0        1.0    3.0
    4                 0          2021                  0          368          2021                  0     0        1.0   10.0
    ...             ...           ...                ...          ...           ...                ...   ...        ...    ...
    176233         3920          2021                  1         3911          2021                  1     2       -1.0    8.0
    176234         3920          2021                  1         3914          2021                  1     1       -1.0    7.0
    176235         3920          2021                  1         3916          2021                  1     3       -1.0    6.0
    176236         3920          2021                  1         3918          2021                  1     0       -1.0    2.0
    176237         3920          2021                  1         3919          2021                  1     3       -1.0    2.0
    
    [176238 rows x 9 columns]
            SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                 0          2021                  0           82          2021                  0     2        1.0    1.0
    1                 0          2021                  0          104          2021                  0     0        1.0    2.0
    2                 0          2021                  0          123          2021                  0     3        1.0    2.0
    3                 0          2021                  0          155          2021                  0     0        1.0    3.0
    4                 0          2021                  0          368          2021                  0     0        1.0   10.0
    ...             ...           ...                ...          ...           ...                ...   ...        ...    ...
    172635         3854          2021                  0         3416          2021                  0     1       -1.0   13.0
    172636         3854          2021                  0         3547          2021                  0     2       -1.0    9.0
    172637         3854          2021                  0         3549          2021                  0     1       -1.0    9.0
    172638         3854          2021                  0         3604          2021                  0     2       -1.0    8.0
    172639         3854          2021                  0         3649          2021                  0     0       -1.0    8.0
    
    [169284 rows x 9 columns]



```python
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
```


```python
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
```


```python
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

      for key, graph in graphs.items():
        print(f"{key}", sep=", ", end="")
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
      fold_model.update_best(avg_oof_loss)
      print(f"      fold {fold_n}: in-fold loss={avg_fold_loss:.4f}, oof loss={avg_oof_loss:.4f}")

      if (epoch_n % (n_epochs // 100) == 0 or
          epoch_n > (n_epochs - 3) or
          fold_model.patience_count > patience - 5):
        fold_time = (datetime.now() - fold_start).total_seconds()
        print(
          f"    epoch {epoch_n:>6}, fold {fold_n}: "
          f"train={avg_fold_loss:.4f} "
          f"valid={avg_oof_loss:.4f} "
          f"patience={fold_model.patience_count}/{patience} "
          f"time={fold_time:.1f}s"
        )

      if fold_model.patience_count > patience:
        print(f"    fold {fold_n} out of patience: valid={fold_model.best_loss:.4f}")

    if all_folds_done:
      print(f"All folds done at epoch {epoch_n}")
      break

    epoch_time = (datetime.now() - epoch_start).total_seconds()
    if epoch_n % (n_epochs // 100) == 0:
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
```


```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
n_features = list(graphs.values())[0].train[0].loc[:, "Men":].shape[1]
layer_sizes = [n_features, 16, 8, 1]
heads = 2
state_dicts = train_kfold_models(kfold, layer_sizes, heads)
```

    Creating fold indices...
      (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
    Iterating over epochs...
      epoch 1
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 2
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 3
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 4
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 5
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 6
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 3
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 4
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 5
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
      epoch 7
        fold 1
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')(2012, 'Women', 'asc')(2012, 'Women', 'des')(2013, 'Men', 'asc')(2013, 'Men', 'des')(2013, 'Women', 'asc')(2013, 'Women', 'des')(2014, 'Men', 'asc')(2014, 'Men', 'des')(2014, 'Women', 'asc')(2014, 'Women', 'des')(2015, 'Men', 'asc')(2015, 'Men', 'des')(2015, 'Women', 'asc')(2015, 'Women', 'des')(2016, 'Men', 'asc')(2016, 'Men', 'des')(2016, 'Women', 'asc')(2016, 'Women', 'des')(2017, 'Men', 'asc')(2017, 'Men', 'des')(2017, 'Women', 'asc')(2017, 'Women', 'des')(2018, 'Men', 'asc')(2018, 'Men', 'des')(2018, 'Women', 'asc')(2018, 'Women', 'des')(2019, 'Men', 'asc')(2019, 'Men', 'des')(2019, 'Women', 'asc')(2019, 'Women', 'des')(2020, 'Men', 'asc')(2020, 'Men', 'des')(2020, 'Women', 'asc')(2020, 'Women', 'des')(2021, 'Men', 'asc')(2021, 'Men', 'des')(2021, 'Women', 'asc')(2021, 'Women', 'des')(2022, 'Men', 'asc')(2022, 'Men', 'des')(2022, 'Women', 'asc')(2022, 'Women', 'des')(2023, 'Men', 'asc')(2023, 'Men', 'des')(2023, 'Women', 'asc')(2023, 'Women', 'des')(2024, 'Men', 'asc')(2024, 'Men', 'des')(2024, 'Women', 'asc')(2024, 'Women', 'des')(2025, 'Men', 'asc')(2025, 'Men', 'des')(2025, 'Women', 'asc')(2025, 'Women', 'des')
        fold 2
          (2003, 'Men', 'asc')(2003, 'Men', 'des')(2004, 'Men', 'asc')(2004, 'Men', 'des')(2005, 'Men', 'asc')(2005, 'Men', 'des')(2006, 'Men', 'asc')(2006, 'Men', 'des')(2007, 'Men', 'asc')(2007, 'Men', 'des')(2008, 'Men', 'asc')(2008, 'Men', 'des')(2009, 'Men', 'asc')(2009, 'Men', 'des')(2010, 'Men', 'asc')(2010, 'Men', 'des')(2010, 'Women', 'asc')(2010, 'Women', 'des')(2011, 'Men', 'asc')(2011, 'Men', 'des')(2011, 'Women', 'asc')(2011, 'Women', 'des')(2012, 'Men', 'asc')(2012, 'Men', 'des')


```python
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
```


```python
# test_models(layer_sizes, state_dicts)
```


```python

```
