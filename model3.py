import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.model_selection import KFold
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

# Step 3: Prepare the Data
edges = list(G.edges(data=True))
edge_labels = [edge[2]['label'] for edge in edges]


positive_edges = [edge for edge in edges if edge[2]['label'] == 1]
negative_edges = [edge for edge in edges if edge[2]['label'] == 0]


if len(positive_edges) < len(negative_edges):
    positive_edges = positive_edges * (len(negative_edges) // len(positive_edges))
else:
    negative_edges = negative_edges * (len(positive_edges) // len(negative_edges))

balanced_edges = positive_edges + negative_edges


node_indices = list(G.nodes())
node_features = torch.zeros(len(node_indices), 3)  # 3 features: degree, clustering coefficient, and eigenvector centrality
for i, node in enumerate(node_indices):
    node_features[i, 0] = G.degree[node]  # Degree
    node_features[i, 1] = nx.clustering(G, node)  # Clustering coefficient
    node_features[i, 2] = nx.eigenvector_centrality(G)[node]  # Eigenvector centrality


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Define the GNN Model
class GNNModel(nn.Module):

    def __init__(self):

        super(GNNModel, self).__init__()



        self.convs = nn.Sequential(

            GCNConv(3, 64),  

            nn.ReLU(),

            nn.Dropout(p=0.5), 

            GCNConv(64, 128), 

            nn.ReLU(),

            nn.Dropout(p=0.5), 

            GCNConv(128, 128),

            nn.ReLU(),

            nn.Dropout(p=0.5)  

        )

        


        self.fc1 = nn.Linear(128, 64) 

        self.fc2 = nn.Linear(64, 2)     # Output layer for binary classification


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.convs[0](x, edge_index)  # First convolution
        x = self.convs[1](x)              # ReLU
        x = self.convs[2](x)              # Dropout
        x = self.convs[3](x, edge_index)  # Second convolution
        x = self.convs[4](x)              # ReLU
        x = self.convs[5](x)              # Dropout
        x = self.convs[6](x, edge_index)  # Third convolution
        x = self.convs[7](x)              # ReLU
        x = self.convs[8](x)              # Dropout
  
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  
        
        return x


# Step 5: Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(kf.split(balanced_edges)):
    print(f'Fold {fold + 1}')
    
  
    train_edges = [balanced_edges[i] for i in train_index]
    val_edges = [balanced_edges[i] for i in val_index]

    train_labels = [edge_labels[edges.index(edge)] for edge in train_edges]
    val_labels = [edge_labels[edges.index(edge)] for edge in val_edges]
    

    id_to_index = {id_: index for index, id_ in enumerate(node_indices)}
    train_edge_indices = [(id_to_index[edge[0]], id_to_index[edge[1]]) for edge in train_edges]
    val_edge_indices = [(id_to_index[edge[0]], id_to_index[edge[1]]) for edge in val_edges]

   
    train_edge_index_tensor = torch.tensor(train_edge_indices).t().contiguous()
    val_edge_index_tensor = torch.tensor(val_edge_indices).t().contiguous()

    train_data = Data(x=node_features, edge_index=train_edge_index_tensor)
    val_data = Data(x=node_features, edge_index=val_edge_index_tensor)


    model = GNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

   
    for epoch in range(20): 
        model.train()
        optimizer.zero_grad()
        
     
        out = model(train_data) 
        edge_node_indices = train_edge_index_tensor[0] 
        out_edges = out[edge_node_indices]  
        
        loss = criterion(out_edges, train_labels_tensor) 
        loss.backward()
        optimizer.step()
        
        
        _, predicted = out_edges.max(dim=1)  # Get the predicted class
        correct = (predicted == train_labels_tensor).sum().item()  # Count correct predictions
        accuracy = correct / train_labels_tensor.size(0)  # Calculate accuracy
        
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

    
    model.eval()
    predicted_labels = []
    predicted_probs = []

    for edge in val_edges:
        node_indices_current = [node_indices.index(edge[0]), node_indices.index(edge[1])]
        node_features_current = node_features[node_indices_current]  

       
        edge_index_current = torch.tensor([[0], [1]]) 

       
        data_current = Data(x=node_features_current.unsqueeze(0), 
                            edge_index=edge_index_current) 

        with torch.no_grad():
            out = model(data_current)
            predicted_label = out.argmax(dim=1)  
            predicted_prob = F.softmax(out, dim=1) 

         
            if predicted_label.numel() > 1:
                predicted_labels.extend(predicted_label.tolist()) 
                predicted_probs.extend(predicted_prob.tolist())  
            else:
                predicted_labels.append(predicted_label.item())  
                predicted_probs.append(predicted_prob.item()) 


    print("test labels ",val_labels)
    
    val_labels_tensor = np.array(val_labels)
    predicted_labels = np.array(predicted_labels)
    binary_predicted_labels=np.argmax(predicted_labels,axis=1)
    print("predicted labels ",binary_predicted_labels)


    precision = precision_score(val_labels_tensor, binary_predicted_labels)
    recall = recall_score(val_labels_tensor, binary_predicted_labels)
    f1 = f1_score(val_labels_tensor, binary_predicted_labels)

    print(f'Fold {fold + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

print("Cross-validation completed.")