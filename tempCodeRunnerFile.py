features = []
# for protein in unique_proteins:
#     if protein in protein_sequences:
#         # Create a tensor for the one-hot encoded features
#         feature_tensor = one_hot_encode(protein_sequences[protein])
#         features.append(feature_tensor)
#     else:

#         # If a protein is missing, add a default feature vector (zero vector)

#         default_feature_tensor = torch.zeros(20)  # Assuming 20 features for one-hot encoding

#         features.append(default_feature_tensor)    

# # Convert the list of features to a tensor
# features_tensor = torch.stack(features)

# # Assign the features tensor to the entire node type
# data['protein'].x = features_tensor  # This sets the x attribute for all protein nodes


# # After creating edge indices, check their maximum values
# # print(f"Max index in positive edge index: {positive_edge_index.max().item()}")
# # print(f"Max index in negative edge index: {negative_edge_index.max().item()}")
# # print(f"Size of node feature tensor: {data['protein'].x.size(0)}")
# # Define the GNN model
# class GNNModel(torch.nn.Module):
#     def __init__(self):
#         super(GNNModel, self).__init__()
#         self.conv1 = GCNConv(20, 64)
#         self.conv2 = GCNConv(64, 32)
#         self.conv3 = GCNConv(32, 16)  # Additional layer
#         self.fc = torch.nn.Linear(16, 1)

#     def forward(self, data):
#         x = data['protein'].x
#         edge_index = data['protein', 'interacts', 'protein'].edge_index
        
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)  # Pass through additional layer
#         x = F.relu(x)

#         row, col = edge_index
#         edge_features = x[row] + x[col]
#         output = self.fc(edge_features)
#         return torch.sigmoid(output)
# # Prepare data for training
# # Create labels for positive and negative interactions
# labels = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
# edges = torch.cat([positive_edge_index, negative_edge_index], dim=1)
# # Debugging: Check the sizes of positive and negative edges
# # print(f"Number of positive edges: {len(positive_edges)}")
# # print(f"Number of negative edges: {len(negative_edges)}")

# # # Check the edge indices
# # print(f"Positive edge index shape: {positive_edge_index.shape}")
# # print(f"Negative edge index shape: {negative_edge_index.shape}")

# # # Check the contents of the edge indices
# # print(f"Positive edge index: {positive_edge_index}")
# # print(f"Negative edge index: {negative_edge_index}")

# # Initialize KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Store metrics
# precision_list = []
# recall_list = []
# f1_list = []

# # Debugging: Check the number of unique proteins and edge indices
# # for protein, index in protein_to_index.items():
# #     print(f"Protein: {protein}, Index: {index}")

# # # Check the edge index for out-of-bounds indices
# # print(f"Edge index: {positive_edge_index}")
# # print(f"Max index in edge index: {positive_edge_index.max().item()}")
# # print(f"Min index in edge index: {positive_edge_index.min().item()}")

# # Perform cross-validation
# for train_index, test_index in kf.split(edges.t().numpy()):
#     # Split the data
#     train_edges = edges[:, train_index]
#     test_edges = edges[:, test_index]
#     train_labels = labels[train_index]
#     test_labels = labels[test_index]

#     # Create training and test data
#     train_data = HeteroData()
#     train_data['protein'].x = data['protein'].x  # Accessing the features correctly
#     train_data['protein', 'interacts', 'protein'].edge_index = train_edges  # Assign edge index for interactions

#     test_data = HeteroData()
#     test_data['protein'].x = data['protein'].x  # Accessing the features correctly
#     test_data['protein', 'interacts', 'protein'].edge_index = test_edges  # Assign edge index for interactions

#     # Initialize model, optimizer, and loss function
#     model = GNNModel()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
#     loss_fn = torch.nn.BCELoss()

#     # Train the model
#     model.train()
#     for epoch in range(100):  # You can adjust the number of epochs
#         optimizer.zero_grad()  # Zero the parameter gradients
        
#         # Forward pass
#         output = model(train_data)
        
#         # Ensure the output size matches the label size
#         if output.size(0) != train_labels.size(0):
#             raise ValueError(f"Output size {output.size(0)} does not match label size {train_labels.size(0)}")

#         # Calculate loss
#         loss = loss_fn(output.view(-1), train_labels)
#         loss.backward()  # Backward pass
#         optimizer.step()  # Optimize the model parameters

#         # Calculate accuracy
#         predicted_labels = (output.view(-1) > 0.5).float()
#         accuracy = (predicted_labels == train_labels).float().mean().item()

#         # Print loss and accuracy for the epoch
#         print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

#     # Evaluate the model
#     model.eval()
#     with torch.no_grad():
#         test_output = model(test_data)
#         predicted_labels = (test_output.view(-1) > 0.5).float()

#         # Calculate metrics
#         precision = precision_score(test_labels.numpy(), predicted_labels.numpy())
#         recall = recall_score(test_labels.numpy(), predicted_labels.numpy())
#         f1 = f1_score(test_labels.numpy(), predicted_labels.numpy())

#         # Append metrics to lists
#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

# # Calculate average metrics
# average_precision = sum(precision_list) / len(precision_list)
# average_recall = sum(recall_list) / len(recall_list)
# average_f1 = sum(f1_list) / len(f1_list)

# # Print results
# print(f'Average Precision: {average_precision:.4f}')
# print(f'Average Recall: {average_recall:.4f}')
# print(f'Average F1 Score: {average_f1:.4f}')