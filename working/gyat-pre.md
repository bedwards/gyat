```python
!pip3 install torch_geometric
```

    Collecting torch_geometric
      Downloading torch_geometric-2.6.1-py3-none-any.whl.metadata (63 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/63.1 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.1/63.1 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.11.14)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2025.3.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.1.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.0.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (5.9.5)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.2.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.32.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (4.67.1)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (6.2.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (0.3.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.18.3)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch_geometric) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2025.1.31)
    Downloading torch_geometric-2.6.1-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m17.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: torch_geometric
    Successfully installed torch_geometric-2.6.1



```python
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.colab import drive

warnings.filterwarnings("ignore")

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.min_rows", 10)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", None)

sns.set_theme(style="whitegrid")

drive_path = "/content/drive"
drive.mount(drive_path)
base_path = f"{drive_path}/My Drive/Colab Notebooks/gyat/input"
data_path = f"{base_path}/march-machine-learning-mania-2025"
gyat_path = f"{base_path}/gyat-dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"
```

    Mounted at /content/drive



```python
sea = []

for gender in ["M", "W"]:
  sea_ = pd.read_csv(f"{data_path}/{gender}Seasons.csv", usecols=["Season", "DayZero"])
  sea_["DayZero"] = pd.to_datetime(sea_["DayZero"])
  sea_ = sea_.rename(columns={"DayZero": f"{gender}DayZero"})
  sea.append(sea_)

sea = pd.merge(sea[0], sea[1], on="Season", how="outer")
sea = sea.sort_values("Season").reset_index(drop=True)

print(f"sea {sea.shape}")
print(sea)
print()
sea.info()
```

    sea (41, 3)
        Season   MDayZero   WDayZero
    0     1985 1984-10-29        NaT
    1     1986 1985-10-28        NaT
    2     1987 1986-10-27        NaT
    3     1988 1987-11-02        NaT
    4     1989 1988-10-31        NaT
    ..     ...        ...        ...
    36    2021 2020-11-02 2020-11-02
    37    2022 2021-11-01 2021-11-01
    38    2023 2022-10-31 2022-10-31
    39    2024 2023-11-06 2023-11-06
    40    2025 2024-11-04 2024-11-04
    
    [41 rows x 3 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41 entries, 0 to 40
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   Season    41 non-null     int64         
     1   MDayZero  41 non-null     datetime64[ns]
     2   WDayZero  28 non-null     datetime64[ns]
    dtypes: datetime64[ns](2), int64(1)
    memory usage: 1.1 KB



```python
tea = pd.DataFrame()

for gender in ["M", "W"]:
  tea = pd.concat([
      tea,
      pd.read_csv(f"{data_path}/{gender}Teams.csv", usecols=["TeamID", "TeamName"]),
  ])

tea = tea.sort_values("TeamID").reset_index(drop=True)

print(f"tea {tea.shape}")
# print(tea)
```

    tea (758, 2)



```python
nodes = []

for gender in ["M", "W"]:
  for part in ["RegularSeason", "NCAATourney"]:
    nodes_gender_part = pd.read_csv(f"{data_path}/{gender}{part}DetailedResults.csv")
    nodes_gender_part["Men"] = gender == "M"
    nodes_gender_part["NCAATourney"] = part == "NCAATourney"
    nodes.append(nodes_gender_part)

nodes = pd.concat(nodes)
nodes["WLoc"] = nodes["WLoc"].map({"A": -1, "N": 0, "H": 1})
nodes["LLoc"] = nodes["WLoc"] * -1

for c in nodes:
  nodes[c] = nodes[c].astype("int32")

both = nodes[[c for c in nodes if c[0] not in ("W", "L")]]

def extract(W_or_L, Le_or_Ri):
  return nodes[[c for c in nodes if c[0] == W_or_L]].rename(columns={c: f"{Le_or_Ri}_{c[1:]}" for c in nodes})

nodes = pd.concat([
  pd.concat([both, extract("W", "Le"), extract("L", "Ri")], axis=1),
  pd.concat([both, extract("L", "Le"), extract("W", "Ri")], axis=1),
])

# Date
nodes = pd.merge(nodes, sea, on="Season")
daynum = pd.to_timedelta(nodes["DayNum"], unit="D")
nodes["Date"] = nodes["WDayZero"] + daynum
nodes.loc[nodes["Men"], "Date"] = nodes["MDayZero"] + daynum
nodes = nodes.drop(columns=["MDayZero", "WDayZero"])

# TeamName
def add_team_name(Le_or_Ri):
  return pd.merge(nodes, tea, left_on=f"{Le_or_Ri}_TeamID", right_on="TeamID"
    ).rename(columns={"TeamName": f"{Le_or_Ri}_TeamName"}
    ).drop(columns=["TeamID"])
nodes = add_team_name("Le")
nodes = add_team_name("Ri")

# Le_Margin
nodes["Le_Margin"] = nodes["Le_Score"] - nodes["Ri_Score"]

# SeasonsAgo
nodes["SeasonsAgo"] = 2025 - nodes["Season"]

# Le_Loc
nodes = nodes.drop(columns=["Ri_Loc"])

# Split ascending and descending TeamIDs
#   so model doesn't learn from noise in arbitrary order
ascending = nodes["Le_TeamID"] < nodes["Ri_TeamID"]
nodes_asc = nodes[ascending]
nodes_des = nodes[~ascending]
del nodes

# Key
def key(ascending=True):
  if ascending:
    df = nodes_asc
    lesser = "Le"
    greater = "Ri"
  else:
    df = nodes_des
    lesser = "Ri"
    greater = "Le"
  df["Key"] = (
    df["Season"].astype(str) + "_" +
    df["DayNum"].astype(str).str.zfill(3) + "_" +
    df[f"{lesser}_TeamID"].astype(str) + "_" +
    df[f"{greater}_TeamID"].astype(str)
  )
key()
key(ascending=False)


def order_columns(df):
  cols = (
    ["Key", "Season", "Date"] +
    ["Le_TeamID", "Ri_TeamID", "Le_TeamName", "Ri_TeamName"] +
    ["Le_Margin"] +
    ["Men", "NCAATourney", "Le_Loc"] +  # features (not scaled)
    ["SeasonsAgo", "DayNum", "NumOT"]  # features (scaled)
  )
  return df[cols + [c for c in df if c not in cols]]


nodes_asc = order_columns(nodes_asc)
nodes_des = order_columns(nodes_des)

print(f"nodes_asc {nodes_asc.shape}")
print(nodes_asc)
print()

print(f"nodes_des {nodes_des.shape}")
print(nodes_des)
```

    nodes_asc (202866, 42)
                           Key  Season       Date  Le_TeamID  Ri_TeamID  Le_TeamName     Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    0       2003_010_1104_1328    2003 2002-11-14       1104       1328      Alabama        Oklahoma          6    1            0       0          22      10      0        68      27      58        3       14      11      18     14     24      13     23       7       1     22        62      22      53        2       10      16      22     10     22       8     18       9       2     20
    1       2003_010_1272_1393    2003 2002-11-14       1272       1393      Memphis        Syracuse          7    1            0       0          22      10      0        70      26      62        8       20      10      19     15     28      16     13       4       4     18        63      24      67        6       24       9      20     20     25       7     12       8       6     16
    2       2003_011_1266_1437    2003 2002-11-15       1266       1437    Marquette       Villanova         12    1            0       0          22      11      0        73      24      58        8       18      17      29     17     26      15     10       5       2     25        61      22      73        3       26      14      23     31     22       9     12       2       5     23
    3       2003_011_1296_1457    2003 2002-11-15       1296       1457   N Illinois        Winthrop          6    1            0       0          22      11      0        56      18      38        3        9      17      31      6     19      11     12      14       2     18        50      18      49        6       22       8      15     17     20       9     19       4       3     23
    6       2003_012_1161_1236    2003 2002-11-16       1161       1236  Colorado St             PFW         18    1            0       1          22      12      0        80      23      55        2        8      32      39     13     18      14     17      11       1     25        62      19      41        4       15      20      28      9     21      11     30      10       4     28
    ...                    ...     ...        ...        ...        ...          ...             ...        ...  ...          ...     ...         ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    405724  2024_145_3124_3425    2024 2024-03-30       3124       3425       Baylor             USC         -4    0            1      -1           1     145      0        70      27      70        9       26       7      12     10     26      17      9       5       4     20        74      26      66        5       17      17      21      6     31      11      9       2       7     11
    405726  2024_146_3333_3376    2024 2024-03-31       3333       3376    Oregon St  South Carolina        -12    0            1       0           1     146      0        58      20      55        8       25      10      13      5     25      17     10       3       5     17        70      26      78        4       20      14      18     17     27      12      6       6       6     14
    405729  2024_151_3163_3234    2024 2024-04-05       3163       3234  Connecticut            Iowa         -2    0            1       0           1     151      0        69      29      63        8       25       3       4      6     22      21     14      15       1     18        71      27      59        7       25      10      14      9     23      12     16       7       1      9
    405730  2024_151_3301_3376    2024 2024-04-05       3301       3376     NC State  South Carolina        -19    0            1       0           1     151      0        59      20      62        6       23      13      18     10     18       5     12       9       1      9        78      33      66        8       19       4       4     10     34      18     15      10       6     16
    405731  2024_153_3234_3376    2024 2024-04-07       3234       3376         Iowa  South Carolina        -12    0            1       0           1     153      0        75      25      63        9       23      16      20      5     20      13      9       6       4     14        87      35      73        8       19       9      17     17     32      16     12       8       8     17
    
    [202866 rows x 42 columns]
    
    nodes_des (202866, 42)
                           Key  Season       Date  Le_TeamID  Ri_TeamID  Le_TeamName   Ri_TeamName  Le_Margin  Men  NCAATourney  Le_Loc  SeasonsAgo  DayNum  NumOT  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    4       2003_011_1208_1400    2003 2002-11-15       1400       1208        Texas       Georgia          6    1            0       0          22      11      0        77      30      61        6       14      11      13     17     22      12     14       4       4     20        71      24      62        6       16      17      27     21     15      12     10       7       1     14
    5       2003_011_1186_1458    2003 2002-11-15       1458       1186    Wisconsin  E Washington         26    1            0       1          22      11      0        81      26      57        6       12      23      27     12     24      12      9       9       3     18        55      20      46        3       11      12      17      6     22       8     19       4       3     25
    8       2003_012_1156_1194    2003 2002-11-16       1194       1156  FL Atlantic  Cleveland St          5    1            0       0          22      12      0        71      28      58        5       11      10      18      9     22       9     17       9       2     23        66      24      52        6       18      12      27     13     26      13     25       8       2     18
    9       2003_012_1296_1458    2003 2002-11-16       1458       1296    Wisconsin    N Illinois         28    1            0       1          22      12      0        84      32      67        5       17      15      19     14     22      11      6      12       0     13        56      23      52        3       14       7      12      9     23      10     18       1       3     18
    11      2003_013_1106_1202    2003 2002-11-17       1202       1106       Furman    Alabama St          1    1            0       0          22      13      0        74      29      51        7       13       9      11      6     21      18     15       7       1      5        73      29      63       10       22       5       5     13     16      15     12       6       2     12
    ...                    ...     ...        ...        ...        ...          ...           ...        ...  ...          ...     ...         ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    405721  2024_145_3163_3181    2024 2024-03-30       3181       3163         Duke   Connecticut         -8    0            1      -1           1     145      0        45      18      55        4       19       5       7     12     25       8     22       6       6     20        53      22      55        3       16       6      10      5     21      16     12      14       6     11
    405723  2024_145_3261_3417    2024 2024-03-30       3417       3261         UCLA           LSU         -9    0            1       1           1     145      0        69      25      70        7       32      12      18     15     27      16     19       6       4     25        78      26      57        2        5      24      31      6     29      11     13      11       5     20
    405725  2024_146_3301_3400    2024 2024-03-31       3400       3301        Texas      NC State        -10    0            1       0           1     146      0        66      29      73        1        6       7      11     16     24      14     13       3       3     21        76      24      53        9       18      19      25      3     23      16     11       6       7     14
    405727  2024_147_3163_3425    2024 2024-04-01       3425       3163          USC   Connecticut         -7    0            1       1           1     147      0        73      23      70        9       29      18      20     10     25      10      9       6       4     20        80      28      58        7       15      17      27      5     30      17     12       6       5     21
    405728  2024_147_3234_3261    2024 2024-04-01       3261       3234          LSU          Iowa         -7    0            1      -1           1     147      0        87      34      88        8       24      11      17     21     28      15     13       6       6     21        94      32      69       13       31      17      22      3     29      16     11       6       3     15
    
    [202866 rows x 42 columns]



```python
def as_struct(nodes):
  cols = ["Index", "Date", "Season", "NCAATourney"]
  return np.array(
    list(nodes[cols].itertuples(index=False)),
    dtype=[(c, nodes[c].dtype) for c in cols]
  )


def create_edges(source, target, edge_type):
  source, target = [pd.DataFrame(n.flatten()) for n in np.meshgrid(source, target)]
  edges = pd.concat([
      source.rename(columns={c: f"Source{c}" for c in source}),
      target.rename(columns={c: f"Target{c}" for c in target}),
  ], axis=1)
  edges["Type"] = edge_type
  edges["Type"] = edges["Type"].astype("int32")
  edges["Delta"] = ((edges["TargetDate"] - edges["SourceDate"]).dt.days).astype("int32")
  edges.insert(9, "Direction", np.sign(edges["Delta"]).astype("int32"))
  edges["Delta"] = np.abs(edges["Delta"])
  edges = edges.drop(columns=["SourceDate", "TargetDate"])
  return edges


def index(df):
  df = df.sort_values("Key").reset_index(drop=True)
  df.index = df.index.astype("int32")
  return df.reset_index(names=["Index"])


for season, men in nodes_asc.groupby(["Season", "Men"]).size().sort_index().reset_index()[["Season", "Men"]].itertuples(index=False):
  gender = "Men" if men else "Women"
  print(f"Processing {season} {gender}")

  nodes_season = nodes_asc[
    (nodes_asc["Season"] == season) &
    (nodes_asc["Men"] == (gender == "Men"))
  ]

  nodes_season = index(nodes_season)
  nodes_season.to_csv(f"{gyat_path}/{season}_{gender}_nodes_asc.csv", index=False)

  nodes_season_des = nodes_des[nodes_des.index.isin(nodes_season.index)]
  nodes_season_des = index(nodes_season_des)
  nodes_season_des.to_csv(f"{gyat_path}/{season}_{gender}_nodes_des.csv", index=False)

  edges = []

  for Le_TeamID, Le_nodes in nodes_season.groupby("Le_TeamID"):
    Le_struct = as_struct(Le_nodes)
    edges.append(create_edges(Le_struct, Le_struct, 0))
    Ri_nodes = nodes_season[nodes_season["Ri_TeamID"] == Le_TeamID]
    Ri_struct = as_struct(Ri_nodes)
    edges.append(create_edges(Le_struct, Ri_struct, 1))

    for Ri_TeamID in Le_nodes["Ri_TeamID"].unique():
      opp_Le_nodes = nodes_season[nodes_season["Le_TeamID"] == Ri_TeamID]
      opp_Le_struct = as_struct(opp_Le_nodes)
      edges.append(create_edges(Le_struct, opp_Le_struct, 4))
      opp_Ri_nodes = nodes_season[(nodes_season["Ri_TeamID"] == Ri_TeamID) & (nodes_season["Le_TeamID"] != Le_TeamID)]
      opp_Ri_struct = as_struct(opp_Ri_nodes)
      edges.append(create_edges(Le_struct, opp_Ri_struct, 5))

  for Ri_TeamID, Ri_nodes in nodes_season.groupby("Ri_TeamID"):
    Ri_struct = as_struct(Ri_nodes)
    edges.append(create_edges(Ri_struct, Ri_struct, 2))
    Le_nodes = nodes_season[nodes_season["Le_TeamID"] == Ri_TeamID]
    Le_struct = as_struct(Le_nodes)
    edges.append(create_edges(Ri_struct, Le_struct, 3))

    for Le_TeamID in Ri_nodes["Le_TeamID"].unique():
      opp_Ri_nodes = nodes_season[nodes_season["Ri_TeamID"] == Le_TeamID]
      opp_Ri_struct = as_struct(opp_Ri_nodes)
      edges.append(create_edges(Ri_struct, opp_Ri_struct, 6))
      opp_Le_nodes = nodes_season[(nodes_season["Le_TeamID"] == Le_TeamID) & (nodes_season["Ri_TeamID"] != Ri_TeamID)]
      opp_Le_struct = as_struct(opp_Le_nodes)
      edges.append(create_edges(Ri_struct, opp_Le_struct, 7))

  edges = pd.concat(edges)
  edges = edges[edges["SourceIndex"] != edges["TargetIndex"]]
  edges = edges.sort_values(["SourceIndex", "TargetIndex", "Type"]).reset_index(drop=True)
  edges.index = edges.index.astype("int32")
  edges.to_csv(f"{gyat_path}/{season}_{gender}_edges.csv", index=False)

  if season in (2006, 2007):
    print(f"edges {edges.shape}")
    print(edges)
    print()
    edges.info()
    print()
```

    Processing 2003 Men
    Processing 2004 Men
    Processing 2005 Men
    Processing 2006 Men
    edges (4002127, 9)
             SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                  0          2006                  0            1          2006                  0     5          0      0
    1                  0          2006                  0            1          2006                  0     7          0      0
    2                  0          2006                  0            3          2006                  0     0          1      1
    3                  0          2006                  0            3          2006                  0     7          1      1
    4                  0          2006                  0            4          2006                  0     2          1      1
    ...              ...           ...                ...          ...           ...                ...   ...        ...    ...
    4002122         4820          2006                  1         4818          2006                  1     0         -1      2
    4002123         4820          2006                  1         4818          2006                  1     7         -1      2
    4002124         4820          2006                  1         4819          2006                  1     2         -1      2
    4002125         4820          2006                  1         4819          2006                  1     4         -1      2
    4002126         4820          2006                  1         4819          2006                  1     5         -1      2
    
    [4002127 rows x 9 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 4002127 entries, 0 to 4002126
    Data columns (total 9 columns):
     #   Column             Dtype
    ---  ------             -----
     0   SourceIndex        int32
     1   SourceSeason       int32
     2   SourceNCAATourney  int32
     3   TargetIndex        int32
     4   TargetSeason       int32
     5   TargetNCAATourney  int32
     6   Type               int32
     7   Direction          int32
     8   Delta              int32
    dtypes: int32(9)
    memory usage: 152.7 MB
    
    Processing 2007 Men
    edges (4762817, 9)
             SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                  0          2007                  0            1          2007                  0     4          0      0
    1                  0          2007                  0            4          2007                  0     0          1      1
    2                  0          2007                  0            4          2007                  0     7          1      1
    3                  0          2007                  0            5          2007                  0     3          1      1
    4                  0          2007                  0            5          2007                  0     4          1      1
    ...              ...           ...                ...          ...           ...                ...   ...        ...    ...
    4762812         5106          2007                  1         5103          2007                  1     7         -1      8
    4762813         5106          2007                  1         5104          2007                  1     0         -1      2
    4762814         5106          2007                  1         5104          2007                  1     7         -1      2
    4762815         5106          2007                  1         5105          2007                  1     2         -1      2
    4762816         5106          2007                  1         5105          2007                  1     5         -1      2
    
    [4762817 rows x 9 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 4762817 entries, 0 to 4762816
    Data columns (total 9 columns):
     #   Column             Dtype
    ---  ------             -----
     0   SourceIndex        int32
     1   SourceSeason       int32
     2   SourceNCAATourney  int32
     3   TargetIndex        int32
     4   TargetSeason       int32
     5   TargetNCAATourney  int32
     6   Type               int32
     7   Direction          int32
     8   Delta              int32
    dtypes: int32(9)
    memory usage: 181.7 MB
    
    Processing 2008 Men
    Processing 2009 Men
    Processing 2010 Women
    Processing 2010 Men
    Processing 2011 Women
    Processing 2011 Men
    Processing 2012 Women
    Processing 2012 Men
    Processing 2013 Women
    Processing 2013 Men
    Processing 2014 Women
    Processing 2014 Men
    Processing 2015 Women
    Processing 2015 Men
    Processing 2016 Women
    Processing 2016 Men
    Processing 2017 Women
    Processing 2017 Men
    Processing 2018 Women
    Processing 2018 Men
    Processing 2019 Women
    Processing 2019 Men
    Processing 2020 Women
    Processing 2020 Men
    Processing 2021 Women
    Processing 2021 Men
    Processing 2022 Women
    Processing 2022 Men
    Processing 2023 Women
    Processing 2023 Men
    Processing 2024 Women
    Processing 2024 Men
    Processing 2025 Women
    Processing 2025 Men



```python

```
