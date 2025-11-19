import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
data_path = os.path.join(os.path.dirname(__file__), '../data/universe_data.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} samples")

# Prepare Data
particles = []
for pid in df['particle_id'].unique():
    pdata = df[df['particle_id'] == pid].sort_values('t')
    coords = pdata[['t', 'x', 'y']].values
    particles.append(coords)

all_coords = np.concatenate(particles)
mean = all_coords.mean(axis=0)
std = all_coords.std(axis=0)
std[std == 0] = 1.0

print(f"Data Mean: {mean}, Std: {std}")

def normalize(coords):
    return (coords - mean) / std

# Create triplets (prev, curr, next)
triplets = []
for p in particles:
    p_norm = normalize(p)
    for i in range(len(p_norm) - 2):
        triplets.append(p_norm[i:i+3])

triplets = np.array(triplets)
triplets_tensor = torch.FloatTensor(triplets).to(DEVICE)
print(f"Training triplets: {len(triplets)}")

# SIREN Layer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MetricNetwork(nn.Module):
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=3):
        super().__init__()
        
        layers = []
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=30))
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=30))
            
        self.net = nn.Sequential(*layers)
        # Output 6 components for symmetric 3x3 metric (g00, g01, g02, g11, g12, g22)
        self.final_linear = nn.Linear(hidden_features, 6)
        
        # Initialize to Minkowski Metric (-1, 1, 1)
        # g00=-1, g01=0, g02=0, g11=1, g12=0, g22=1
        with torch.no_grad():
            self.final_linear.weight.fill_(0.0)
            self.final_linear.bias.data = torch.tensor([-1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        
    def forward(self, coords):
        # coords: (Batch, 3)
        x = self.net(coords)
        out = self.final_linear(x)
        
        # Construct symmetric matrix
        # g00, g01, g02, g11, g12, g22
        g00 = out[:, 0]
        g01 = out[:, 1]
        g02 = out[:, 2]
        g11 = out[:, 3]
        g12 = out[:, 4]
        g22 = out[:, 5]
        
        # Stack into (Batch, 3, 3)
        # Row 0
        row0 = torch.stack([g00, g01, g02], dim=1)
        # Row 1
        row1 = torch.stack([g01, g11, g12], dim=1)
        # Row 2
        row2 = torch.stack([g02, g12, g22], dim=1)
        
        g = torch.stack([row0, row1, row2], dim=1)
        return g

model = MetricNetwork().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def get_christoffel(coords):
    # coords: (Batch, 3)
    # We need gradients of g w.r.t coords
    # Efficient batch gradient computation
    
    coords.requires_grad_(True)
    g = model(coords) # (Batch, 3, 3)
    g_inv = torch.inverse(g)
    
    # Compute derivatives dg_mu_nu / dx_sigma
    # We can use torch.autograd.grad with sum trick
    
    batch_size = coords.shape[0]
    Dg = torch.zeros(batch_size, 3, 3, 3).to(DEVICE)
    
    # Unique indices for symmetric matrix: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    indices = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    
    for (mu, nu) in indices:
        # Grad of sum(g[:, mu, nu]) w.r.t coords
        # grad output is (Batch, 3) -> dg_mu_nu / dx_sigma
        grad = torch.autograd.grad(g[:, mu, nu].sum(), coords, create_graph=True)[0]
        Dg[:, mu, nu, :] = grad
        Dg[:, nu, mu, :] = grad # Symmetric
            
    # Gamma^k_ij = 0.5 * g^kl * (dg_jl/dx_i + dg_il/dx_j - dg_ij/dx_l)
    
    # Vectorized einsum
    # Dg is (Batch, mu, nu, sigma) = d_sigma g_mu_nu
    
    # d_i g_jl -> Dg[:, j, l, i] -> permute(0, 2, 3, 1) ? No.
    # Dg indices: 0:Batch, 1:mu, 2:nu, 3:sigma
    # We want d_i g_jl. i=sigma, j=mu, l=nu.
    # So we want Dg[:, j, l, i]
    # Permute Dg(B, mu, nu, sigma) -> (B, sigma, mu, nu) is (B, i, j, l)
    d_i_g_jl = Dg.permute(0, 3, 1, 2)
    
    # d_j g_il -> Dg[:, i, l, j]
    # We want (B, i, j, l) where j=sigma, i=mu, l=nu.
    # Permute Dg(B, mu, nu, sigma) -> (B, mu, sigma, nu) is (B, i, j, l)
    d_j_g_il = Dg.permute(0, 1, 3, 2)
    
    # d_l g_ij -> Dg[:, i, j, l]
    # We want (B, i, j, l) where l=sigma, i=mu, j=nu.
    # Permute Dg(B, mu, nu, sigma) -> (B, mu, nu, sigma) is (B, i, j, l)
    d_l_g_ij = Dg
    
    bracket = d_i_g_jl + d_j_g_il - d_l_g_ij
    
    # Gamma^k_ij = 0.5 * g^kl * bracket_ijl
    # einsum: bkl, bijl -> bkij
    Gamma = 0.5 * torch.einsum('bkl,bijl->bkij', g_inv, bracket)
                
    return Gamma, g

def loss_function(batch_triplets):
    # batch_triplets: (Batch, 3, 3) -> (prev, curr, next)
    
    x_prev = batch_triplets[:, 0, :]
    x_curr = batch_triplets[:, 1, :]
    x_next = batch_triplets[:, 2, :]
    
    # Estimate velocity and acceleration (finite differences)
    u = (x_next - x_prev) / 2.0
    a = (x_next - 2*x_curr + x_prev)
    
    # Compute Gamma at x_curr
    Gamma, g = get_christoffel(x_curr)
    
    # Compute Geodesic Residual
    # Gamma_term^k = Gamma^k_ij * u^i * u^j
    Gamma_term = torch.einsum('bkij,bi,bj->bk', Gamma, u, u)
    
    residual = a + Gamma_term
    
    geodesic_loss = torch.mean(residual**2)
    
    # Regularization
    det = torch.det(g)
    det_loss = torch.mean((torch.abs(det) - 1.0)**2)
    
    # Signature prior
    sig_loss = torch.mean(torch.relu(g[:, 0, 0])) + \
               torch.mean(torch.relu(-g[:, 1, 1])) + \
               torch.mean(torch.relu(-g[:, 2, 2]))
               
    return geodesic_loss + 0.1 * det_loss + 0.1 * sig_loss

# Training
epochs = 1000
batch_size = 256
losses = []

print("Training Metric Network (Robust)...")
for epoch in range(epochs):
    indices = torch.randperm(len(triplets_tensor))
    epoch_loss = 0
    batches = 0
    
    for i in range(0, len(triplets_tensor), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = triplets_tensor[batch_idx]
        
        optimizer.zero_grad()
        loss = loss_function(batch)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        batches += 1
        
    avg_loss = epoch_loss / batches
    losses.append(avg_loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

# Save Model
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '../models/siren_metric.pth'))
print("Saved model to models/siren_metric.pth")

# Visualization
r_vals = np.linspace(0.1, 8.0, 100)
g00_vals = []
g11_vals = []

model.eval()
with torch.no_grad():
    for r in r_vals:
        pt = np.array([0.0, r, 0.0])
        pt_norm = normalize(pt)
        pt_tensor = torch.FloatTensor([pt_norm]).to(DEVICE)
        
        g = model(pt_tensor)[0]
        g00_vals.append(g[0, 0].item())
        g11_vals.append(g[1, 1].item())

plt.figure(figsize=(10, 5))
plt.plot(r_vals, g00_vals, label='g_00 (Time)')
plt.plot(r_vals, g11_vals, label='g_11 (Radial)')
plt.axvline(x=5.0, color='k', linestyle='--', label='Star Surface')
plt.xlabel('Radius r')
plt.ylabel('Metric Component')
plt.title('Learned Metric Components')
plt.legend()
plt.grid(True)
plt.savefig('learned_metric.png')
print("Saved learned_metric.png")
