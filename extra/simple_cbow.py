import torch
import torch.nn as nn
import torch.optim as optim
from functools import lru_cache
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Toy corpus
sentences = [
    "le chien aboie fort dans la maison",
    "le chien aboie dans le jardin",
    "le grand chien aboie dans le jardin",
    "le petit chien aboie dans la maison",
    "le chien fatigué dort dans la maison",
    "le vieux chien dort dans le jardin",
    "le chien affamé mange dans la maison",
    "le chien joyeux joue dans le jardin",
    "le chat gris dort dans la maison",
    "le chat noir dort dans le jardin",
    "le jeune chat miaule dans le jardin",
    "le petit chat miaule dans la maison",
    "le chat siamois miaule dans le jardin",
    "le gros chat miaule dans la maison",
    "le chien aboie quand chat miaule",
    "le chat miaule quand chien aboie",
    "le chien court quand chat dort",
    "le chat dort quand chien joue",
]
sentence_words = [sentence.split() for sentence in sentences]

# Parameters
CONTEXT_SIZE = 2  # 2 words to the left and right
EMBEDDING_DIM = 10  # Size of the word embeddings

# Create a flat corpus for vocabulary building
corpus = [word for sentence in sentence_words for word in sentence]

# Vocabulary
vocab = list(set(corpus))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(vocab)


# Create CBOW context vector
@lru_cache(maxsize=None)
def make_context_vector(context):
    """
    Convert a tuple of context words to a tensor of indices.

    Args:
        context: Tuple of context words (must be hashable for caching)

    Returns:
        Tensor of word indices
    """
    return torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)


# Create CBOW training data, ensuring context words come from the same sentence
data = []
for sentence in sentence_words:
    if len(sentence) >= 2 * CONTEXT_SIZE + 1:  # Only process sentences with enough words
        for i in range(CONTEXT_SIZE, len(sentence) - CONTEXT_SIZE):
            context = []
            for j in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
                if j != 0:  # Skip the target word (j=0)
                    context.append(sentence[i + j])
            target = sentence[i]
            data.append((context, target))

print(f"Created {len(data)} training examples")


# CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the CBOW model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embeddings
        """
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        """
        Forward pass of the CBOW model.

        Args:
            context_idxs: Tensor of context word indices

        Returns:
            Output logits for the vocabulary
        """
        embeds = self.embeddings(context_idxs)
        mean_embed = embeds.mean(dim=0)
        out = self.linear(mean_embed)
        return out


# Initialize model on the correct device
model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-4)

# Training loop
print(f"Starting training with {len(data)} examples")
for epoch in range(3000):
    total_loss = 0
    for context, target in data:
        # Convert the context list to a tuple for caching
        context_idxs = make_context_vector(tuple(context))
        target_tensor = torch.tensor([word_to_ix[target]])

        # Forward pass
        model.zero_grad()
        logits = model(context_idxs)

        # Compute loss
        loss = loss_function(logits.view(1, -1), target_tensor)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# Extract the learned embeddings and visualize them
def plot_embeddings(embeddings, ix_to_word):
    """
    Plot and save the word embeddings in 2D space.

    Args:
        embeddings: 2D array of reduced embeddings
        ix_to_word: Dictionary mapping indices to words
    """
    plt.figure(figsize=(12, 10))

    # Plot all word embeddings
    for i, word in ix_to_word.items():
        x, y = embeddings[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), textcoords="offset points", xytext=(0, 5), ha="center")

    # Draw vectors between related words
    word_pairs = [("chat", "miaule"), ("chien", "aboie")]
    for word1, word2 in word_pairs:
        if word1 in word_to_ix and word2 in word_to_ix:
            x1, y1 = embeddings[word_to_ix[word1]]
            x2, y2 = embeddings[word_to_ix[word2]]

            # Calculate arrow length to avoid overlapping with points
            arrow_scale = 0.8  # Adjust to leave space at the target point
            dx = (x2 - x1) * arrow_scale
            dy = (y2 - y1) * arrow_scale

            plt.arrow(x1, y1, dx, dy, head_width=0.2, head_length=0.4, fc="magenta", ec="magenta", width=0.01)

    plt.title("CBOW Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("cbow_embeddings.png")
    print("Embeddings visualization saved to 'cbow_embeddings.png'")


# Get the learned embeddings and reduce dimensions for visualization
with torch.no_grad():
    # First get embeddings from model and move to CPU for NumPy processing
    learned_embeddings = model.embeddings.weight.cpu().numpy()

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(learned_embeddings)

    # Plot the embeddings
    plot_embeddings(reduced_embeddings, ix_to_word)
