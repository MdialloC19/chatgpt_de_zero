from .dataset import Dataset
from collections import Counter
import re
from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size=10000, min_frequency=2):
        """Initialize the BPE tokenizer."""
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = {}

    def _pretokenize(self, text):
        """Pre-tokenize the text to handle punctuation and spaces."""
        text = text.lower()  # Convert to lowercase, because our dataset is lowercase only

        # This regex splits on word boundaries but keeps punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]|\s", text)
        return tokens

    def train(self, dataset: Dataset):
        """Train the BPE tokenizer on the dataset."""
        # Step 1: Create initial vocabulary with special tokens
        vocab = {"<UNK>": 0, "<PAD>": 1, "<SPACE>": 2}
        idx = len(vocab)

        # Step 2: Tokenize text properly, keeping punctuation as separate tokens
        all_tokens = []
        for text in dataset.data:
            # This regex splits on word boundaries but keeps punctuation as separate tokens
            tokens = self._pretokenize(text)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Step 3: Initialize each token as character sequence
        token_splits = {}
        for token, count in token_counts.items():
            if count >= self.min_frequency:
                if token == " ":
                    token_splits["<SPACE>"] = ["<SPACE>"]  # Treat space as a special token
                else:
                    token_splits[token] = list(token)
                # Add each character to vocab if not already there
                for char in token:
                    if char not in vocab:
                        vocab[char] = idx
                        idx += 1

        # Step 4: BPE algorithm - merge most frequent pairs
        num_merges = max(self.vocab_size - len(vocab), 0)

        for _ in tqdm(range(num_merges)):
            # Count all symbol pairs
            pairs = Counter()
            for token, freq in token_counts.items():
                if token in token_splits:
                    symbols = token_splits[token]
                    for i in range(len(symbols) - 1):
                        pairs[symbols[i], symbols[i + 1]] += freq

            # Break if no more pairs
            if not pairs:
                break

            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break

            # Create new symbol
            new_symbol = "".join(best_pair)

            # Add to vocabulary
            vocab[new_symbol] = idx
            idx += 1

            # Apply merge to all splits
            for token in token_splits:
                symbols = token_splits[token]
                i = 0
                while i < len(symbols) - 1:
                    if (symbols[i], symbols[i + 1]) == best_pair:
                        symbols[i] = new_symbol
                        symbols.pop(i + 1)
                    else:
                        i += 1

            # Check if we've reached vocab size
            if len(vocab) >= self.vocab_size:
                break

        self.vocab = vocab

    def encode(self, text):
        """Encode text using BPE with Maximum Matching First (greedy) algorithm."""
        # Tokenize the text properly
        tokens = self._pretokenize(text)
        token_ids = []

        for token in tokens:
            # Handle space token specially
            if token == " ":
                token_ids.append(self.vocab.get("<SPACE>", self.vocab["<UNK>"]))
                continue

            # Check if the entire token exists in vocabulary
            if token in self.vocab:
                token_ids.append(self.vocab[token])
                continue

            # Maximum Matching First algorithm for unknown tokens
            i = 0
            current_token_ids = []

            while i < len(token):
                # Try to find the longest matching subtokens
                best_len = 0
                best_match = None

                # Look for the longest possible match starting at position i
                for end in range(len(token), i, -1):
                    subtoken = token[i:end]
                    if subtoken in self.vocab:
                        best_len = end - i
                        best_match = subtoken
                        break

                # If no match found, use single character and mark as unknown if needed
                if best_match is None:
                    char = token[i]
                    current_token_ids.append(self.vocab.get(char, self.vocab["<UNK>"]))
                    i += 1
                else:
                    # Add the matching token
                    current_token_ids.append(self.vocab[best_match])
                    i += best_len

            token_ids.extend(current_token_ids)

        return token_ids

    def get_tokens_with_values(self, text):
        """Get tokens with their values from the text."""
        tokens = self.encode(text)
        tokens_with_values = []
        for token in tokens:
            token_value = next((k for k, v in self.vocab.items() if v == token), "<UNK>")
            tokens_with_values.append((token_value, token))
        return tokens_with_values

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        # Create reverse vocabulary for lookup
        id_to_token = {v: k for k, v in self.vocab.items()}

        # Convert IDs to tokens
        tokens = [id_to_token.get(id, "<UNK>") for id in token_ids]

        # Join tokens - this is simplistic and might need refinement
        # for proper handling of whitespace around punctuation
        text = "".join(tokens).replace("<SPACE>", " ")
        return text

    def save(self, path):
        """Save the tokenizer to a file."""
        import json

        data = {
            "vocab": self.vocab,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, path):
        """Load the tokenizer from a file."""
        import json
        import ast

        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
