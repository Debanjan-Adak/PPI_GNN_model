import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load the Data
positive_interactions = pd.read_csv('positive_sample.csv', header=None, names=['protein1', 'protein2'])
negative_interactions = pd.read_csv('negative_sample.csv', header=None, names=['protein1', 'protein2'])

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
node_features = torch.randn(len(node_indices), 128)  # Random node features
edge_features = torch.randn(len(edge_indices), 128)  # Random edge features

# Step 5: Create a mapping from string identifiers to unique integers
id_to_index = {id_: index for index, id_ in enumerate(node_indices)}

# Step 6: Convert edges to their corresponding indices
edge_indices = [(id_to_index[edge[0]], id_to_index[edge[1]]) for edge in edge_indices]

# Step 7: Create the tensor
edge_index_tensor = torch.tensor(edge_indices).t().contiguous()

# Create the Data object
data = Data(x=node_features, edge_index=edge_index_tensor, edge_attr=edge_features)

# Step 8: Define the GNN Model
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Step 9: Train the GNN Model
model = GNNModel()
criterion = nn.CrossEntropyLoss()

# Convert train_labels to a tensor and ensure it has the correct shape
train_labels_tensor = torch.tensor(train_labels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    
    # Ensure the output corresponds to the training edges
    out = out[edge_index_tensor[0]]  # Select only the output for the nodes involved in the edges
    
    # Compute loss
    loss = criterion(out, train_labels_tensor)  # Ensure labels are in tensor format
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = out.max(dim=1)  # Get the predicted class
    correct = (predicted == train_labels_tensor).sum().item()  # Count correct predictions
    accuracy = correct / train_labels_tensor.size(0)  # Calculate accuracy
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

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

# Print the predicted labels for the test edges
#print("Predicted labels for test edges:", predicted_labels)