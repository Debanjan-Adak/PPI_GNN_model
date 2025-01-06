import networkx as nx
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
# Step 1: Load the Data
positive_interactions = pd.read_csv('positive_100.csv', header=None, names=['protein1', 'protein2'])
negative_interactions = pd.read_csv('negative_100.csv', header=None, names=['protein1', 'protein2'])

print("Step 1 Done")
# Step 2: Create the Interaction Graph
G = nx.Graph()
proteins = set(list(positive_interactions['protein1']) + list(positive_interactions['protein2']) +
                list(negative_interactions['protein1']) + list(negative_interactions['protein2']))
G.add_nodes_from(proteins)

for index, row in positive_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=1)

for index, row in negative_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=0)
    
print("Step 2 Done")
# Step 3: Split the Data
edges = list(G.edges(data=True))
edge_labels = [edge[2]['label'] for edge in edges]

# Oversample the minority class
positive_edges = [edge for edge in edges if edge[2]['label'] == 1]
negative_edges = [edge for edge in edges if edge[2]['label'] == 0]

# Balance the dataset
if len(positive_edges) < len(negative_edges):
    positive_edges = positive_edges * (len(negative_edges) // len(positive_edges))
else:
    negative_edges = negative_edges * (len(positive_edges) // len(negative_edges))

balanced_edges = positive_edges + negative_edges
train_edges, test_edges, train_labels, test_labels = train_test_split(balanced_edges, edge_labels, test_size=0.2, random_state=42)

print("Step 3 Done")
# Step 4: Convert the Graph to a GNN Format
node_indices = list(G.nodes())
edge_indices = list(train_edges)  # Use the original edges from the training set

# Compute topological features
node_features = torch.zeros(len(node_indices), 3)  # 3 features: degree, clustering coefficient, and eigenvector centrality
for i, node in enumerate(node_indices):
    node_features[i,0 ] = G.degree[node]  # Degree
    node_features[i, 1] = nx.clustering(G, node)  # Clustering coefficient
    node_features[i, 2] = nx.degree_assortativity_coefficient(G)  # Eigenvector centrality
edge_features = torch.randn(len(edge_indices), 128)  # Random edge features

print("Step 4 Done")
# Step 5: Create a mapping from string identifiers to unique integers
id_to_index = {id_: index for index, id_ in enumerate(node_indices)}
print("Step 5 Done")
# Step 6: Convert edges to their corresponding indices
edge_indices = [(id_to_index[edge[0]], id_to_index[edge[1]]) for edge in edge_indices]
print("Step 6 Done")
# Step 7: Create the tensor
edge_index_tensor = torch.tensor(edge_indices).t().contiguous()

# Create the Data object
data = Data(x=node_features, edge_index=edge_index_tensor, edge_attr=edge_features)
print("Step 7 Done")
# Step 8: Define the GNN Model

class GNNModel(nn.Module):

    def __init__(self):

        super(GNNModel, self).__init__()

        

        # Define the convolutional layers

        self.conv1 = GCNConv(3, 64)  # Input features are 3 (topological features), output features are 64

        self.conv2 = GCNConv(64, 128)  # Second layer with 128 output features

        self.conv3 = GCNConv(128, 128)  # Third layer with 128 output features

        

        # Fully connected layers

        self.fc1 = nn.Linear(128, 64)  # Fully connected layer

        self.fc2 = nn.Linear(64, 2)     # Output layer for binary classification


    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        

        # Pass through the convolutional layers

        x = self.conv1(x, edge_index)  # First convolution

        x = F.relu(x)                  # Apply ReLU activation

        x = F.dropout(x, p=0.5)        # Apply dropout

        

        x = self.conv2(x, edge_index)  # Second convolution

        x = F.relu(x)                  # Apply ReLU activation

        x = F.dropout(x, p=0.5)        # Apply dropout

        

        x = self.conv3(x, edge_index)  # Third convolution

        x = F.relu(x)                  # Apply ReLU activation

        x = F.dropout(x, p=0.5)        # Apply dropout

        

        # No global pooling, return node-level features

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)  # Output layer

        

        return x

print("Step 8 Done")
# Step 9: Train the GNN Model
model = GNNModel()
criterion = nn.CrossEntropyLoss()

# Convert train_labels to a tensor and ensure it has the correct shape
train_labels_tensor = torch.tensor(train_labels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
# Step 9: Train the GNN Model
model = GNNModel()
criterion = nn.CrossEntropyLoss()

# Convert train_labels to a tensor and ensure it has the correct shape and type
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)  # Ensure labels are Long type

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):  # Increased from 20 to 100
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)  # Get the output for all nodes
    
    # Select the output for the nodes involved in the edges
    edge_node_indices = edge_index_tensor[0]  # Get the source nodes for the edges
    out_edges = out[edge_node_indices]  # Select the output for the source nodes
    
    # Compute loss
    loss = criterion(out_edges, train_labels_tensor)  # Ensure labels are in tensor format
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = out_edges.max(dim=1)  # Get the predicted class
    correct = (predicted == train_labels_tensor).sum().item()  # Count correct predictions
    accuracy = correct / train_labels_tensor.size(0)  # Calculate accuracy
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')
print("Step 9 Done")

print("Type of test_labels:", type(test_labels))

# Step 10: Evaluate the GNN Model
model.eval()
predicted_labels = []

for edge in test_edges:
    node_indices_current = [node_indices.index(edge[0]), node_indices.index(edge[1])]
    node_features_current = node_features[node_indices_current]  # Extract node features for the current edge
    
    # Create edge_index for the current edge (2x1 tensor)
    edge_index_current = torch.tensor([[0], [1]])  # Edge index for the current edge
    
    # Create the Data object for the current edge
    data_current = Data(x=node_features_current.unsqueeze(0),  # Add batch dimension
                        edge_index=edge_index_current)  # Use the current edge index
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        out = model(data_current)
        predicted_label = out.argmax(dim=1)  # Get the predicted class as a tensor
        
        # Check if predicted_label is a tensor with more than one element
        if predicted_label.numel() > 1:
            predicted_labels.extend(predicted_label.tolist())  # Convert to list and extend
        else:
            predicted_labels.append(predicted_label.item())  # Convert to scalar and append
            
print("Type of predicted_labels:", type(predicted_labels))

binary_predicted_labels = np.argmax(predicted_labels, axis=1)
print("Content of test_labels:", test_labels)
print("Content of original predicted_labels:", predicted_labels)
print("Content of predicted_labels:", binary_predicted_labels)
# Calculate evaluation metrics
precision = precision_score(test_labels, binary_predicted_labels)
recall = recall_score(test_labels, binary_predicted_labels)
f1 = f1_score(test_labels, binary_predicted_labels)

#print("Predicted labels for test edges:", predicted_labels)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')