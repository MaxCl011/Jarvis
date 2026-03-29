import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

#device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#dataset class
class JarvisDataset(Dataset):
    def __init__(self, tokens_path):
        self.data = torch.load(tokens_path)
        print(f"Dataset loaded: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
        #input is every token except the last one, target is every token except the first one
        input_ids = self.data[idx][:-1]
        labels = self.data[idx][1:]
        return input_ids, labels

#load dataset and dataloader
Dataset = JarvisDataset("tokenised_data.pt")
DataLoader = DataLoader(Dataset, batch_size=4, shuffle=True)

print(f"Batches per epoch: {len(DataLoader):,}")

#load GPT-2 model

print("\nLoading GPT-2 model...")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded. Total parameters: {total_params:,}")


#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#training loop
Num_Epochs = 3
print(f"\nStarting training for {Num_Epochs} epochs...")

for epoch in range(Num_Epochs):

    model.train()
    total_loss = 0
    num_batches = 0
   
    for batch_idx, batch in enumerate(DataLoader):

        # Move batch to device
        input_ids = batch.to(device)
        labels = batch.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        #Gradient clipping (optional but can help with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1:>2}/{Num_Epochs} | Average Loss: {avg_loss:.4f}")        


print("\nTraining complete! Saving model and tokenizer...")


#save model and tokenizer
save_path = "jarvis_model"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved to '{save_path}'")