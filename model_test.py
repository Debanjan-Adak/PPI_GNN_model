import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Load the Data
positive_interactions = pd.read_csv('positive_100.csv', header=None, names=['protein1', 'protein2'])
negative_interactions = pd.read_csv('negative_100.csv', header=None, names=['protein1', 'protein2'])

print("Step 1 Done")

# Create the Interaction Graph
G = nx.Graph()
proteins = set(list(positive_interactions['protein1']) + list(positive_interactions['protein2']) +
                list(negative_interactions['protein1']) + list(negative_interactions['protein2']))
G.add_nodes_from(proteins)

for index, row in positive_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=1)

for index, row in negative_interactions.iterrows():
    G.add_edge(row['protein1'], row['protein2'], label=0)

print("Step 2 Done")

# Step 3: Split the Data and Implement 5-Fold Cross-Validation
edges = list(G.edges(data=True))
edge_labels = [edge[2]['label'] for edge in edges]

# Oversample the minority class
positive_edges = [edge for edge in edges if edge[2]['label'] == 1]
negative_edges = [edge for edge in edges if edge[2]['label'] == 0]

# Combine edges if they are balanced
if len(positive_edges) == len(negative_edges):
    balanced_edges = positive_edges + negative_edges
else:
    raise ValueError("The dataset is not balanced.")

# Prepare for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []

# Step 4: Convert the Graph to a GNN Format
node_indices = list(G.nodes())
node_features = torch.randn(len(node_indices), 128)  # Random node features
print("Step 4 Done")

# Step 5: Define the GNN Model
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

print("Step 5 Done")

# Step 6: Train the GNN Model
model = GNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Store metrics for each fold
fold_metrics = []

for train_index, test_index in kf.split(balanced_edges):
    train_edges = [balanced_edges[i] for i in train_index]
    test_edges = [balanced_edges[i] for i in test_index]
    
    train_labels = [edge[2]['label'] for edge in train_edges]
    test_labels = [edge[2]['label'] for edge in test_edges]

    # Convert train_labels to a tensor
    train_labels_tensor = torch.tensor(train_labels)

    # Prepare the data for training
    edge_indices = [(node_indices.index(edge[0]), node_indices.index(edge[1])) for edge in train_edges]
    edge_index_tensor = torch.tensor(edge_indices).t().contiguous()
    data = Data(x=node_features, edge_index=edge_index_tensor)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Ensure the output corresponds to the training edges
        out = out[edge_index_tensor[0]]
        
        # Compute loss
        loss = criterion(out, train_labels_tensor)  # Ensure labels are in tensor format
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = out.max(dim=1)  # Get the predicted class
        correct = (predicted == train_labels_tensor).sum().item()  # Count correct predictions
        accuracy = correct / train_labels_tensor.size(0)  # Calculate accuracy
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

    # Evaluate the model on the test set
    model.eval()
    predicted_labels = []
    predicted_probs = []  # To store predicted probabilities

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
            predicted_prob = F.softmax(out, dim=1)  # Get predicted probabilities
            
            # Check if predicted_label is a tensor with more than one element
            if predicted_label.numel() > 1:
                predicted_labels.extend(predicted_label.tolist())  # Convert to list and extend
                predicted_probs.extend(predicted_prob.tolist())  # Convert to list and extend
            else:
                predicted_labels.append(predicted_label.item())  # Convert to scalar and append
                predicted_probs.append(predicted_prob.item())  # Convert to scalar and append

    # print(f"Number of test labels: {len(test_labels)}")

    # print(f"Number of predicted labels: {len(predicted_labels)}")

    # print(f"Number of predicted probabilities: {len(predicted_probs)}")
    # print(f"Number of predicted probabilities: {predicted_probs}")
    # Convert predicted_labels and predicted_probs to numpy arrays
    #print("test labels ",test_labels)
    #print("predicted labels ",predicted_labels)
    predicted_labels = np.array(predicted_labels)
    binary_predicted_labels=np.argmax(predicted_labels,axis=1)
    #print("binary labewls labels ",binary_predicted_labels)
    #predicted_probs = np.array(predicted_probs)[:, 1]  # Get probabilities for the positive class
    predicted_probs.extend(predicted_prob[:, 1].tolist())  # Extract probabilities for the positive class
    predicted_probs = predicted_probs[:-1]  # This will keep all elements except the last one
    # print(f"Number of predicted probabilities: {len(predicted_probs)}")
    # print(f"Number of predicted probabilities: {predicted_probs}")
    
    # Calculate evaluation metrics
    precision = precision_score(test_labels, binary_predicted_labels)
    recall = recall_score(test_labels, binary_predicted_labels)
    f1 = f1_score(test_labels, binary_predicted_labels)
    auc = roc_auc_score(test_labels, binary_predicted_labels)

    fold_metrics.append((precision, recall, f1, auc))
    print(f'Fold Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, predicted_probs)
    pr_auc = auc(recall_curve, precision_curve)

    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, color='green', label='AUPRC (area = {:.2f})'.format(pr_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

# Step 7: Average metrics across folds
avg_metrics = np.mean(fold_metrics, axis=0)
print(f'Average Metrics - Precision: {avg_metrics[0]:.4f}, Recall: {avg_metrics[1]:.4f}, F1 Score: {avg_metrics[2]:.4f}, AUC: {avg_metrics[3]:.4f}')

# Step 8: Save the Model
joblib.dump(model, 'gnn_model.joblib')
print("Model saved as gnn_model.joblib")
torch.save(model.state_dict(), 'gnnmodel.pth')
print("Saved gnn model as torch model")

# Step 9: Load the Model (Optional)
loaded_model = GNNModel()
loaded_model.load_state_dict(torch.load('gnnmodel.pth', weights_only=True))
loaded_model.eval()  # Set the model to evaluation mode

print("Model loaded from gnn_model.pth")