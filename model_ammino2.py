import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

# Load positive and negative interaction data
positive_df = pd.read_csv('positive_100.csv', header=None)  # No header
negative_df = pd.read_csv('negative_100.csv', header=None)  # No header

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

# Add nodes to the HeteroData object
for protein in unique_proteins:
    data['protein'][protein] = {}

# Add edges for positive interactions using indices
positive_edges = [(protein_to_index[row[0]], protein_to_index[row[1]]) for _, row in positive_df.iterrows()]
positive_edge_index = torch.tensor(positive_edges, dtype=torch.long).t().contiguous()
data['protein', 'interacts', 'protein'].edge_index = positive_edge_index  # Correct assignment

# Add edges for negative interactions using indices
negative_edges = [(protein_to_index[row[0]], protein_to_index[row[1]]) for _, row in negative_df.iterrows()]
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

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(20, 64)  # Assuming 20 features from one-hot encoding
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)  # Binary classification

    def forward(self, data):
        x = data['protein'].x
        edge_index = data['protein', 'interacts', 'protein'].edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Gather edge features for the output
        row, col = edge_index
        edge_features = x[row] + x[col]  # Combine features of connected nodes
        output = self.fc(edge_features)  # Pass through a fully connected layer
        return torch.sigmoid(output)  # Apply sigmoid to get probabilities
# Prepare data for training
# Create labels for positive and negative interactions
labels = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
edges = torch.cat([positive_edge_index, negative_edge_index], dim=1)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
precision_list = []
recall_list = []
f1_list = []

# Perform cross-validation
for train_index, test_index in kf.split(edges.t().numpy()):
    # Split the data
    train_edges = edges[:, train_index]
    test_edges = edges[:, test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]

    # Create training and test data
    train_data = HeteroData()
    train_data['protein'].x = data['protein'].x  # Accessing the features correctly
    train_data['protein', 'interacts', 'protein'].edge_index = train_edges  # Assign edge index for interactions

    test_data = HeteroData()
    test_data['protein'].x = data['protein'].x  # Accessing the features correctly
    test_data['protein', 'interacts', 'protein'].edge_index = test_edges  # Assign edge index for interactions

    # Initialize model, optimizer, and loss function
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    # Print the train_data structure for debugging
    print(train_data)

    # Train the model
    model.train()
    for epoch in range(100):  # You can adjust the number of epochs
        optimizer.zero_grad()
        output = model(train_data)
        # print(f"Output size: {output.size()}")
        # print(f"Train labels size: {train_labels.size()}")
        
        # Ensure the output size matches the label size
        if output.size(0) != train_labels.size(0):
            raise ValueError(f"Output size {output.size(0)} does not match label size {train_labels.size(0)}")

        loss = loss_fn(output.view(-1), train_labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_output = model(test_data)
        predicted_labels = (test_output.view(-1) > 0.5).float()

        # Calculate metrics
        precision = precision_score(test_labels.numpy(), predicted_labels.numpy())
        recall = recall_score(test_labels.numpy(), predicted_labels.numpy())
        f1 = f1_score(test_labels.numpy(), predicted_labels.numpy())

        # Append metrics to lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

# Calculate average metrics
average_precision = sum(precision_list) / len(precision_list)
average_recall = sum(recall_list) / len(recall_list)
average_f1 = sum(f1_list) / len(f1_list)

# Print results
print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {average_recall:.4f}')
print(f'Average F1 Score: {average_f1:.4f}')