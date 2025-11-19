import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# Constants
DEVICE = torch.device("cpu") # CPU for precision and autograd safety
G = 1.0
c = 1.0
M = 1.0
R = 5.0

# MetricNetwork Definition (Must match training)
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
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
        self.final_linear = nn.Linear(hidden_features, 6)
        
    def forward(self, coords):
        # coords: (Batch, 3)
        x = self.net(coords)
        out = self.final_linear(x)
        
        # Construct symmetric matrix
        g00 = out[:, 0]
        g01 = out[:, 1]
        g02 = out[:, 2]
        g11 = out[:, 3]
        g12 = out[:, 4]
        g22 = out[:, 5]
        
        row0 = torch.stack([g00, g01, g02], dim=1)
        row1 = torch.stack([g01, g11, g12], dim=1)
        row2 = torch.stack([g02, g12, g22], dim=1)
        
        g = torch.stack([row0, row1, row2], dim=1)
        return g

# Load Model
model_path = os.path.join(os.path.dirname(__file__), '../models/siren_metric.pth')
model = MetricNetwork().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.double() # Use Double Precision
model.eval()

# Data Normalization (Must match training)
data_path = os.path.join(os.path.dirname(__file__), '../data/universe_data.csv')
df = pd.read_csv(data_path)
all_coords = df[['t', 'x', 'y']].values
mean = torch.tensor(all_coords.mean(axis=0), dtype=torch.double).to(DEVICE)
std = torch.tensor(all_coords.std(axis=0), dtype=torch.double).to(DEVICE)
std[std == 0] = 1.0

def normalize(coords):
    return (coords - mean) / std

def get_metric(coords):
    # coords: (1, 3)
    # We need to normalize inside here if the model expects normalized inputs
    # The training fed normalized inputs.
    
    # But wait, derivatives w.r.t real coordinates?
    # If we feed normalized coords, derivatives will be w.r.t normalized coords.
    # We need derivatives w.r.t real coords.
    # dx_norm / dx_real = 1/std
    
    # Let's define a function that takes REAL coords, normalizes them, and returns metric.
    # But the metric itself is a tensor. If we change coordinates, the metric components transform!
    # g_uv(x) = g_ab(xi) * dxi^a/dx^u * dxi^b/dx^v
    # The model outputs components in the NORMALIZED coordinate basis?
    # Or did we treat the model outputs as components in the REAL basis?
    
    # In training:
    # u = (x_next - x_prev) / 2.0 (Normalized difference)
    # a = (x_next - 2*x_curr + x_prev) (Normalized acceleration)
    # Gamma was computed from g(normalized_coords) w.r.t normalized coords.
    # The loss was a + Gamma u u.
    # Everything was in normalized space.
    
    # So the model learned the metric of the "Normalized Universe".
    # We need to calculate curvature in this Normalized Universe.
    # The relationship G_00 = k * rho should still hold, possibly with a scaling factor.
    # Or we can transform everything back to real units.
    
    # Let's calculate G_00 in the normalized space first.
    # The density rho is a scalar, so it's invariant (value at a point).
    
    # So: Input normalized coords -> Output metric in normalized basis.
    # Calculate derivatives w.r.t normalized coords.
    
    # We don't need to normalize inside this function if we pass normalized coords to it.
    # Let's assume 'coords' passed here are already normalized for the autograd to work directly.
    
    return model(coords).squeeze(0)

def get_christoffel(coords):
    # coords: (1, 3) normalized
    
    def metric_func(x):
        return get_metric(x)
    
    # Jacobian of metric w.r.t coords -> (3, 3, 3)
    # Dg[mu, nu, lambda] = dg_mu_nu / dx_lambda
    Dg = torch.autograd.functional.jacobian(metric_func, coords).squeeze()
    
    g = metric_func(coords)
    g_inv = torch.inverse(g)
    
    Gamma = torch.zeros(3, 3, 3, dtype=torch.double).to(DEVICE)
    for sigma in range(3):
        for mu in range(3):
            for nu in range(3):
                val = 0.0
                for rho in range(3):
                    val += 0.5 * g_inv[sigma, rho] * (
                        Dg[nu, rho, mu] + Dg[mu, rho, nu] - Dg[mu, nu, rho]
                    )
                Gamma[sigma, mu, nu] = val
                
    return Gamma, g

def get_riemann_ricci_einstein(coords):
    # coords: (1, 3) normalized
    
    def gamma_func(x):
        G, _ = get_christoffel(x)
        return G
        
    DGamma = torch.autograd.functional.jacobian(gamma_func, coords).squeeze()
    Gamma, g = get_christoffel(coords)
    
    R_tensor = torch.zeros(3, 3, 3, 3, dtype=torch.double).to(DEVICE)
    
    for rho in range(3):
        for sigma in range(3):
            for mu in range(3):
                for nu in range(3):
                    term1 = DGamma[rho, nu, sigma, mu]
                    term2 = DGamma[rho, mu, sigma, nu]
                    
                    term3 = 0.0
                    term4 = 0.0
                    for lam in range(3):
                        term3 += Gamma[rho, mu, lam] * Gamma[lam, nu, sigma]
                        term4 += Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]
                        
                    R_tensor[rho, sigma, mu, nu] = term1 - term2 + term3 - term4
                    
    Ricci = torch.zeros(3, 3, dtype=torch.double).to(DEVICE)
    for mu in range(3):
        for nu in range(3):
            val = 0.0
            for lam in range(3):
                val += R_tensor[lam, mu, lam, nu]
            Ricci[mu, nu] = val
            
    g_inv = torch.inverse(g)
    R_scalar = 0.0
    for mu in range(3):
        for nu in range(3):
            R_scalar += g_inv[mu, nu] * Ricci[mu, nu]
            
    G_tensor = Ricci - 0.5 * R_scalar * g
    
    return G_tensor, g

print("Calculating Curvature (Direct Metric)...")

results = []
r_vals = np.linspace(0.1, 8.0, 50)

for r in r_vals:
    # Real point
    pt_real = np.array([0.0, r, 0.0])
    
    # Normalize for model
    pt_norm = (pt_real - mean.cpu().numpy()) / std.cpu().numpy()
    pt_tensor = torch.tensor([pt_norm], dtype=torch.double, requires_grad=True).to(DEVICE)
    
    try:
        G_tensor, g_tensor = get_riemann_ricci_einstein(pt_tensor)
        G_00 = G_tensor[0, 0].item()
        g_00 = g_tensor[0, 0].item()
        g_11 = g_tensor[1, 1].item()
        
        # Density (Real space)
        if r <= R:
            rho = M / (4/3 * np.pi * R**3)
        else:
            rho = 0.0
            
        results.append({
            'r': r,
            'G_00': G_00,
            'g_00': g_00,
            'g_11': g_11,
            'rho': rho
        })
        
        if len(results) % 5 == 0:
            print(f"r={r:.2f} | g_00={g_00:.4f} | G_00={G_00:.4e} | rho={rho:.4e}")
            
    except Exception as e:
        print(f"Error at r={r}: {e}")

res_df = pd.DataFrame(results)
output_path = os.path.join(os.path.dirname(__file__), '../data/curvature_data.csv')
res_df.to_csv(output_path, index=False)
print(f"Saved curvature data to {output_path}")
