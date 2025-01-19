import matplotlib.pyplot as plt

def load_protein_sequences(file_path):
    protein_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                protein_dict[parts[0]] = parts[1]
    return protein_dict

# Load the protein sequences from the file
protein_sequences = load_protein_sequences('protein_sequences.txt')

# Calculate the lengths of the protein sequences
sequence_lengths = [len(seq) for seq in protein_sequences.values()]

# Create a histogram of the sequence lengths
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Protein Sequence Lengths')
plt.xlabel('Length of Protein Sequences')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.xlim(0, max(sequence_lengths) + 100)  # Adjust x-axis limit for better visibility
plt.show()