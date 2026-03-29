from transformers import GPT2Tokenizer
import torch
import os

#load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")


#load data
def load_text_files(data_dir):
    all_text = ""
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                all_text += content + "\n"
                print(f" Loaded {filename} ({len(content):,} chars)")
    print(f"Total Characters: {len(all_text):,}")
    return all_text

print("\n Loading text files...")
raw_text = load_text_files("data/")


#Tokenise
print("\n Tokenising text...")
all_tokens = tokenizer.encode(raw_text)
print(f"Total tokens: {len(all_tokens):,}")

#Chunk into sequences
CHUNK_SIZE = 256

CHUNKS = []
for i in range(0, len(all_tokens) - CHUNK_SIZE, CHUNK_SIZE):
    CHUNKS.append(all_tokens[i:i+CHUNK_SIZE])

print(f"Total chunks: {len(CHUNKS):,}")

#convert to tensors
chunks_tensor = torch.tensor(CHUNKS, dtype=torch.long)
print(f"Tensor shape: {chunks_tensor.shape}")


#save for use in training
torch.save(chunks_tensor, "tokenised_data.pt")
torch.save(tokenizer, "tokenizer.pt")
print("\n Tokenised data and tokenizer saved.")
print("Ready for training!")