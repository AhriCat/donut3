"""
Donut3 Training Example
========================
Basic training loop for any Donut3 variant.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Simple dataset wrapping tokenized text chunks."""

    def __init__(self, token_ids, seq_len=128):
        self.seq_len = seq_len
        # Chunk into fixed-length sequences
        n = len(token_ids) // seq_len * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x[:-1], "labels": x[1:]}


def train(
    model,
    tokenizer,
    texts,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    seq_len: int = 128,
    device: str = "cpu",
):
    """
    Basic training loop.

    Args:
        model: Any Donut3 variant
        tokenizer: TernaryTokenizer (must be trained and frozen)
        texts: List of strings to train on
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        seq_len: Sequence length for chunking
        device: Device string
    """
    model = model.to(device)
    model.train()

    # Tokenize all texts
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)

    if len(all_ids) < seq_len:
        print(f"Warning: only {len(all_ids)} tokens, need at least {seq_len}")
        return

    dataset = TextDataset(all_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pad_id = getattr(tokenizer, 'PAD_ID', getattr(tokenizer, 'pad_id', 0))

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")


if __name__ == "__main__":
    from model import Donut3
    from tokenizer import TernaryTokenizer

    # Create tokenizer
    tokenizer = TernaryTokenizer(vocab_size=1000)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Intelligence is hierarchical, self-consistent, and adaptive. " * 20,
    ]
    tokenizer.train(sample_texts, min_frequency=1)
    tokenizer.freeze()

    # Create model
    model = Donut3(
        vocab_size=len(tokenizer.token_to_id),
        dim=64,
        depth=2,
        heads=4,
        groups=2,
        rank=16,
        ssm_dim=16,
        rnn_dim=32,
        dropout=0.1,
        max_seq_len=128,
        num_cycloidal_modes=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Donut3 parameters: {total_params:,}")
    print(f"PVA projection compression ratio: {model.projection.pcp.compression_ratio:.1f}x")

    train(model, tokenizer, sample_texts, epochs=3, seq_len=64, batch_size=2)
