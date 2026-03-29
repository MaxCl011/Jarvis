import torch
import torch.nn as nn

# ── Device setup ──────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ── Model definition ──────────────────────────────────────
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


# ── Generate fake training data ───────────────────────────
# In a real project you'd load a CSV here instead
torch.manual_seed(42)   # makes the random data the same every run

num_houses = 200

# 200 houses, 4 features each — random but realistic-ish ranges
bedrooms   = torch.randint(1, 6,    (num_houses, 1)).float()
bathrooms  = torch.randint(1, 4,    (num_houses, 1)).float()
sqft       = torch.randint(500, 4000, (num_houses, 1)).float()
distance   = torch.rand(num_houses, 1) * 20

# Stack into one [200, 4] tensor
X = torch.cat([bedrooms, bathrooms, sqft, distance], dim=1)

# Create fake prices using a simple formula + some noise
# price = 50k*beds + 30k*baths + 100*sqft - 10k*distance + noise
noise = torch.randn(num_houses, 1) * 20000
y = (50000 * bedrooms +
     30000 * bathrooms +
     100   * sqft -
     10000 * distance +
     noise)

# Move data to device
X = X.to(device)
y = y.to(device)

print(f"Training data shape: {X.shape}")   # [200, 4]
print(f"Labels shape:        {y.shape}")   # [200, 1]


# ── Normalise the data ────────────────────────────────────
# Features like sqft (500-4000) are on a totally different scale
# than distance (0-20). This confuses the model. Normalising
# brings everything to a similar range (mean 0, std 1).
X_mean = X.mean(dim=0)
X_std  = X.std(dim=0)
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std  = y.std()
y_norm = (y - y_mean) / y_std


# ── Create model, loss function, optimiser ────────────────
model     = HousePriceModel().to(device)
loss_fn   = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


# ── Training loop ─────────────────────────────────────────
num_epochs = 500

for epoch in range(num_epochs):

    # Put model in training mode
    model.train()

    # 1. Forward pass
    predictions = model(X_norm)

    # 2. Compute loss
    loss = loss_fn(predictions, y_norm)

    # 3. Zero gradients
    optimiser.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Update weights
    optimiser.step()

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:>4}/{num_epochs} | Loss: {loss.item():.6f}")


print("\nTraining complete!")


# ── Test the trained model ────────────────────────────────
model.eval()   # switch to evaluation mode (disables dropout etc.)

with torch.no_grad():   # we don't need gradients for inference
    # A new house: 3 beds, 2 baths, 1500 sqft, 5km from city
    new_house = torch.tensor([[3.0, 2.0, 1500.0, 5.0]]).to(device)

    # Normalise using the same mean/std from training data
    new_house_norm = (new_house - X_mean) / X_std

    # Get prediction (will be normalised)
    pred_norm = model(new_house_norm)

    # Un-normalise back to real dollar values
    pred_price = pred_norm * y_std + y_mean

    print(f"\nPredicted price for new house: ${pred_price.item():,.0f}")

    # What would the formula say? (our "ground truth")
    true_approx = 50000*3 + 30000*2 + 100*1500 - 10000*5
    print(f"Expected price (formula):      ${true_approx:,.0f}")