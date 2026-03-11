import torch
import time
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-medium"

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()

sequence_lengths = [16, 32, 64, 128, 256]
batch_sizes = [1, 2, 4, 8]

results = []

for seq_len in sequence_lengths:

    text = "Hello world " * seq_len
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"][:, :seq_len]

    for batch in batch_sizes:

        batch_input = input_ids.repeat(batch, 1).to(device)

        for _ in range(5):
            _ = model(batch_input)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        runs = 20
        for _ in range(runs):
            _ = model(batch_input)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()

        latency = (end - start) / runs
        tokens_per_sec = (batch * seq_len) / latency

        results.append({
            "model": model_name,
            "sequence_length": seq_len,
            "batch_size": batch,
            "latency_s": latency,
            "tokens_per_sec": tokens_per_sec
        })

df = pd.DataFrame(results)
print(df)

df.to_csv("gpt2_medium_results.csv", index=False)