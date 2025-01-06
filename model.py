import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load the Data
positive_interactions = pd.read_csv('positive_100.csv', header=None, names=['protein1', 'protein2'])
negative_interactions = pd.read_csv('negative_100.csv', header=None, names=['protein1', 'protein2'])

# Step 2: Create the Interaction Graph
G = nx.Graph()
proteins = set(list(positive_interactions['protein1']) + list(positive_interactions['protein2']) +
                list(negative_interactions['protein1']) + list(negative_interactions['protein2']))
G.add_nodes_from(proteins)

for index, row in positive_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=1)

for index, row in negative_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=0)

# Step 3: Split the Data
edges = list(G.edges(data=True))
edge_labels = [edge[2]['label'] for edge in edges]

train_edges, test_edges, train_labels, test_labels = train_test_split(edges, edge_labels, test_size=0.2, random_state=42)

# Step 4: Convert the Graph to a GNN Format
node_indices = list(G.nodes())
edge_indices = list(train_edges)  # Use the original edges from the training set