import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

# Load positive and negative interaction data
positive_df = pd.read_csv('positive_sample.csv', header=None)  # No header
negative_df = pd.read_csv('negative_sample.csv', header=None)  # No header

# Load amino acid sequences
def load_protein_sequences(file_path):
    protein_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                protein_dict[parts[0]] = parts[1]
    return protein_dict

protein_sequences = load_protein_sequences('protein_sequences.txt')

# Create a HeteroData object
data = HeteroData()

# Add nodes for proteins
unique_proteins = set(positive_df[0]).union(set(positive_df[1])).union(set(negative_df[0])).union(set(negative_df[1]))

# Create a mapping from protein names to unique indices
protein_to_index = {protein: idx for idx, protein in enumerate(unique_proteins)}
#print(protein_to_index)
# Add nodes to the HeteroData object
for protein in unique_proteins:
    data['protein'][protein] = {}

# Create edges for positive interactions using indices
positive_edges = []
for _, row in positive_df.iterrows():
    if row[0] in protein_to_index and row[1] in protein_to_index:
        positive_edges.append((protein_to_index[row[0]], protein_to_index[row[1]]))

# Check if any edges are invalid
if not positive_edges:
    raise ValueError("No valid positive edges found.")

positive_edge_index = torch.tensor(positive_edges, dtype=torch.long).t().contiguous()

# Create edges for negative interactions using indices
negative_edges = []
for _, row in negative_df.iterrows():
    if row[0] in protein_to_index and row[1] in protein_to_index:
        negative_edges.append((protein_to_index[row[0]], protein_to_index[row[1]]))

# Check if any edges are invalid
if not negative_edges:
    raise ValueError("No valid negative edges found.")

negative_edge_index = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()
data['protein', 'does_not_interact', 'protein'].edge_index = negative_edge_index  # Correct assignment

# Feature extraction: One-hot encoding of amino acid sequences
def one_hot_encode(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
    encoding = torch.zeros(len(amino_acids), dtype=torch.float)
    for aa in sequence:
        if aa in amino_acids:
            encoding[amino_acids.index(aa)] += 1
    return encoding

# Assign features to nodes
features = []
for protein in unique_proteins:
    if protein in protein_sequences:
        # Create a tensor for the one-hot encoded features
        feature_tensor = one_hot_encode(protein_sequences[protein])
        features.append(feature_tensor)

# Convert the list of features to a tensor
features_tensor = torch.stack(features)

# Assign the features tensor to the entire node type
data['protein'].x = features_tensor  # This sets the x attribute for all protein nodes

# Check the number of unique proteins
print(f"Total unique proteins: {len(unique_proteins)}")

# Check the number of proteins in the protein_sequences dictionary
print(f"Proteins in protein_sequences: {len(protein_sequences)}")

# Identify missing proteins
missing_proteins = [protein for protein in unique_proteins if protein not in protein_sequences]
print(f"Missing proteins: {missing_proteins}")
print(f"Number of missing proteins: {len(missing_proteins)}")