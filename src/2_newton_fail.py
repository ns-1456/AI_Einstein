import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load Data
data_path = os.path.join(os.path.dirname(__file__), '../data/universe_data.csv')
df = pd.read_csv(data_path)

print(f"Loaded data: {len(df)} samples")

# Prepare Data for Training
# Input: Position (x, y)
# Target: Acceleration (ax, ay)
# We need to compute acceleration from the data numerically
# a = dv/dt. Since we have discrete steps, we can approximate.

# Group by particle_id to compute derivatives correctly
inputs = []
targets = []

for pid in df['particle_id'].unique():
    pdata = df[df['particle_id'] == pid].sort_values('t')
    
    # Calculate dt
    dt = pdata['t'].diff().shift(-1)
    
    # Calculate dvx, dvy
    dvx = pdata['vx'].diff().shift(-1)
    dvy = pdata['vy'].diff().shift(-1)
    
    # Calculate acceleration
    ax = dvx / dt
    ay = dvy / dt
    
    # Filter out last row (NaN)
    valid_indices = ~np.isnan(ax)
    
    p_inputs = pdata[['x', 'y']][valid_indices].values
    p_targets = np.stack([ax[valid_indices], ay[valid_indices]], axis=1)
    
    inputs.append(p_inputs)
    targets.append(p_targets)

X = np.concatenate(inputs)
Y = np.concatenate(targets)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(Y)

print(f"Training samples: {len(X_tensor)}")

# Define Newtonian Model (MLP)
# It tries to learn a vector field F(x,y) -> (ax, ay)
# This assumes acceleration depends only on position (conservative force field like Gravity)
class NewtonianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: ax, ay
        )
        
    def forward(self, x):
        return self.net(x)

model = NewtonianModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
epochs = 1000
losses = []

print("Training Newtonian Model...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Visualize Results
# 1. Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Newtonian Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('newton_loss.png')
print("Saved newton_loss.png")

# 2. Newtonian Drift (Prediction vs Reality)
# Let's take one particle and simulate its trajectory using the trained Newtonian model
# vs the actual Relativistic trajectory

test_pid = 0
pdata = df[df['particle_id'] == test_pid].sort_values('t')
t_vals = pdata['t'].values
x_real = pdata['x'].values
y_real = pdata['y'].values
vx_real = pdata['vx'].values
vy_real = pdata['vy'].values

# Simulate Newtonian path
# Initial conditions
x_curr = x_real[0]
y_curr = y_real[0]
vx_curr = vx_real[0]
vy_curr = vy_real[0]

x_pred = [x_curr]
y_pred = [y_curr]

model.eval()
with torch.no_grad():
    for i in range(len(t_vals) - 1):
        dt = t_vals[i+1] - t_vals[i]
        
        # Predict acceleration
        pos_tensor = torch.FloatTensor([[x_curr, y_curr]])
        acc = model(pos_tensor).numpy()[0]
        ax_pred, ay_pred = acc[0], acc[1]
        
        # Update velocity
        vx_curr += ax_pred * dt
        vy_curr += ay_pred * dt
        
        # Update position
        x_curr += vx_curr * dt
        y_curr += vy_curr * dt
        
        x_pred.append(x_curr)
        y_pred.append(y_curr)

    print(f"Real Trajectory Range: X [{x_real.min():.2f}, {x_real.max():.2f}], Y [{y_real.min():.2f}, {y_real.max():.2f}]")
    print(f"Pred Trajectory Range: X [{min(x_pred):.2f}, {max(x_pred):.2f}], Y [{min(y_pred):.2f}, {max(y_pred):.2f}]")

plt.figure(figsize=(10, 10))
# Plot Star
circle = plt.Circle((0, 0), 5.0, color='orange', alpha=0.3, label='Star')
plt.gca().add_patch(circle)

# Plot Real Trajectory
plt.plot(x_real, y_real, 'b-', label='Relativistic Reality (Data)', linewidth=2)

# Plot Newtonian Prediction
plt.plot(x_pred, y_pred, 'r--', label='Newtonian Prediction (Model)', linewidth=2)

plt.title(f'The Newtonian Failure: Particle {test_pid}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.grid(True)
plt.savefig('newtonian_drift.png')
print("Saved newtonian_drift.png")

# Save Model
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '../models/newton_baseline.pth'))
print("Saved model to models/newton_baseline.pth")
