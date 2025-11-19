import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Constants (G=1, c=1)
G = 1.0
c = 1.0
M = 1.0  # Mass of the star
R = 5.0   # Radius of the star
rs = 2 * G * M / c**2  # Schwarzschild radius

print(f"Star Mass: {M}")
print(f"Star Radius: {R}")
print(f"Schwarzschild Radius: {rs}")

if rs >= R:
    raise ValueError("Schwarzschild radius must be smaller than star radius for a stable non-collapsed star.")

def metric_components(r):
    """
    Returns g_tt (A) and g_rr (B) for the Schwarzschild metric.
    Handles both Interior (r <= R) and Exterior (r > R) cases.
    """
    if r <= R:
        # Interior Schwarzschild Solution (Constant Density)
        # A(r) = [3/2 * sqrt(1 - rs/R) - 1/2 * sqrt(1 - rs*r^2/R^3)]^2
        # B(r) = [1 - rs*r^2/R^3]^(-1)
        
        term1 = 1.5 * np.sqrt(1 - rs/R)
        term2 = 0.5 * np.sqrt(1 - rs * r**2 / R**3)
        A = (term1 - term2)**2
        
        B = 1.0 / (1 - rs * r**2 / R**3)
        
        # Density is constant inside
        rho = M / (4/3 * np.pi * R**3)
        
    else:
        # Exterior Schwarzschild Solution
        # A(r) = 1 - rs/r
        # B(r) = (1 - rs/r)^(-1)
        
        A = 1 - rs/r
        B = 1.0 / (1 - rs/r)
        
        # Density is zero outside
        rho = 0.0
        
    return A, B, rho

def derivatives_metric(r):
    """
    Computes dA/dr and dB/dr numerically or analytically.
    Using analytical derivatives for precision.
    """
    epsilon = 1e-5
    if r <= R:
        # Analytical derivatives for Interior
        # Let u = 1 - rs*r^2/R^3
        # du/dr = -2*rs*r/R^3
        # A = (C - 0.5*sqrt(u))^2
        # dA/dr = 2*(C - 0.5*sqrt(u)) * (-0.5 * 1/(2*sqrt(u)) * du/dr)
        #       = (C - 0.5*sqrt(u)) * (-1/(2*sqrt(u))) * (-2*rs*r/R^3)
        #       = (C - 0.5*sqrt(u)) * (rs*r / (R^3 * sqrt(u)))
        
        term1 = 1.5 * np.sqrt(1 - rs/R)
        u = 1 - rs * r**2 / R**3
        sqrt_u = np.sqrt(u)
        
        dA_dr = (term1 - 0.5 * sqrt_u) * (rs * r) / (R**3 * sqrt_u)
        
        # B = 1/u
        # dB/dr = -1/u^2 * du/dr = -1/u^2 * (-2*rs*r/R^3) = 2*rs*r / (R^3 * u^2)
        dB_dr = 2 * rs * r / (R**3 * u**2)
        
    else:
        # Analytical derivatives for Exterior
        # A = 1 - rs/r
        # dA/dr = rs/r^2
        dA_dr = rs / r**2
        
        # B = (1 - rs/r)^(-1) = r / (r - rs)
        # dB/dr = -1 * (1 - rs/r)^(-2) * (rs/r^2) = -rs / (r^2 * (1 - rs/r)^2)
        # Or simpler: B = r/(r-rs). dB/dr = [ (r-rs) - r ] / (r-rs)^2 = -rs / (r-rs)^2
        dB_dr = -rs / ((r - rs)**2) # Wait, let's check.
        # B = (1 - rs/r)^-1
        # dB/dr = -(1 - rs/r)^-2 * (rs/r^2) = - B^2 * (rs/r^2).
        # If B = r/(r-rs), B^2 = r^2/(r-rs)^2.
        # - r^2/(r-rs)^2 * rs/r^2 = -rs/(r-rs)^2. Correct.
        
        # However, standard form B = 1/(1-rs/r).
        # dB/dr = -1/(1-rs/r)^2 * (rs/r^2).
        # This is positive? No. 1-rs/r < 1.
        # Wait. A = 1 - 2M/r. dA/dr = 2M/r^2 > 0.
        # B = (1 - 2M/r)^-1. dB/dr = -1 * (1-2M/r)^-2 * (2M/r^2).
        # This is negative. B goes from infinity at rs to 1 at infinity. It should be decreasing. So negative is correct.
        
        B_val = 1.0 / (1 - rs/r)
        dB_dr = - (B_val**2) * (rs / r**2)

    return dA_dr, dB_dr

def geodesic_equations(tau, state):
    """
    Geodesic equations for Schwarzschild metric in equatorial plane (theta = pi/2).
    State vector: [t, r, phi, dt/dtau, dr/dtau, dphi/dtau]
    """
    t, r, phi, ut, ur, uphi = state
    
    # Avoid singularity
    if r < 0.1:
        return [ut, ur, uphi, 0, 0, 0] # Stop evolution effectively
    
    A, B, _ = metric_components(r)
    dA_dr, dB_dr = derivatives_metric(r)
    
    # Christoffel Symbols (non-zero ones for diagonal metric)
    # Gamma^t_tr = Gamma^t_rt = 1/2A * dA/dr
    # Gamma^r_tt = 1/2B * dA/dr
    # Gamma^r_rr = 1/2B * dB/dr
    # Gamma^r_phiphi = -r/B
    # Gamma^phi_rphi = Gamma^phi_phir = 1/r
    
    Gamma_t_tr = 0.5 / A * dA_dr
    Gamma_r_tt = 0.5 / B * dA_dr
    Gamma_r_rr = 0.5 / B * dB_dr
    Gamma_r_pp = -r / B
    Gamma_p_rp = 1.0 / r
    
    # Geodesic Equations: d2x^u/dtau^2 = - Gamma^u_ab * u^a * u^b
    
    # dt/dtau equation
    # d(ut)/dtau = - (2 * Gamma^t_tr * ut * ur)
    dut_dtau = -2 * Gamma_t_tr * ut * ur
    
    # dr/dtau equation
    # d(ur)/dtau = - (Gamma^r_tt * ut^2 + Gamma^r_rr * ur^2 + Gamma^r_pp * uphi^2)
    dur_dtau = - (Gamma_r_tt * ut**2 + Gamma_r_rr * ur**2 + Gamma_r_pp * uphi**2)
    
    # dphi/dtau equation
    # d(uphi)/dtau = - (2 * Gamma^p_rp * ur * uphi)
    duphi_dtau = -2 * Gamma_p_rp * ur * uphi
    
    return [ut, ur, uphi, dut_dtau, dur_dtau, duphi_dtau]

def generate_data(num_particles=50, t_span=(0, 100)):
    all_data = []
    
    print(f"Generating {num_particles} trajectories...")
    
    for i in range(num_particles):
        # Initial Conditions
        # Start outside the star, moving inwards
        r0 = R * 1.5 + np.random.rand() * R  # Start between 1.5R and 2.5R
        phi0 = np.random.rand() * 2 * np.pi
        
        # Initial velocity (coordinate velocity)
        # Aim towards the center roughly
        v_r = -0.1 - np.random.rand() * 0.2 # Inward radial velocity
        v_phi = (np.random.rand() - 0.5) * 0.1 / r0 # Small angular velocity
        
        # Convert to 4-velocity components
        # We need to satisfy normalization condition: g_uv u^u u^v = 1 (for massive particles)
        # A * (ut)^2 - B * (ur)^2 - r^2 * (uphi)^2 = 1
        
        A, B, _ = metric_components(r0)
        
        # Assume initial coordinate time t=0
        t0 = 0
        
        # Guess ur and uphi from coordinate velocities?
        # v^i = dx^i / dt = (dx^i/dtau) / (dt/dtau) = u^i / ut
        # So u^r = ut * v_r, u^phi = ut * v_phi
        
        # Substitute into normalization:
        # A * ut^2 - B * (ut*v_r)^2 - r0^2 * (ut*v_phi)^2 = 1
        # ut^2 * (A - B*v_r^2 - r0^2*v_phi^2) = 1
        # ut = 1 / sqrt(A - B*v_r^2 - r0^2*v_phi^2)
        
        denom = A - B * v_r**2 - r0**2 * v_phi**2
        if denom <= 0:
            print(f"Skipping particle {i}: Impossible initial velocity (superluminal?)")
            continue
            
        ut0 = 1.0 / np.sqrt(denom)
        ur0 = ut0 * v_r
        uphi0 = ut0 * v_phi
        
        initial_state = [t0, r0, phi0, ut0, ur0, uphi0]
        
        # Solve ODE
        sol = solve_ivp(geodesic_equations, t_span, initial_state, rtol=1e-8, atol=1e-8, max_step=0.1)
        
        # Extract data
        ts = sol.y[0]
        rs = sol.y[1]
        phis = sol.y[2]
        uts = sol.y[3]
        urs = sol.y[4]
        uphis = sol.y[5]
        
        # Convert to Cartesian for output
        xs = rs * np.cos(phis)
        ys = rs * np.sin(phis)
        
        # Coordinate velocities: vx = dx/dt, vy = dy/dt
        # dx/dt = (dx/dr * dr/dt + dx/dphi * dphi/dt) ?? No.
        # vx = d(r cos phi)/dt = dr/dt cos phi - r sin phi dphi/dt
        # dr/dt = ur / ut
        # dphi/dt = uphi / ut
        
        dr_dt = urs / uts
        dphi_dt = uphis / uts
        
        vxs = dr_dt * np.cos(phis) - rs * np.sin(phis) * dphi_dt
        vys = dr_dt * np.sin(phis) + rs * np.cos(phis) * dphi_dt
        
        # Get density at each point
        rhos = []
        for r_val in rs:
            _, _, rho_val = metric_components(r_val)
            rhos.append(rho_val)
        
        # Store data
        # We want (t, x, y, vx, vy, rho)
        # Note: 't' here is coordinate time, which matches the 't' in the solution if we solved for it.
        # Yes, sol.y[0] is t(tau).
        
        particle_df = pd.DataFrame({
            't': ts,
            'x': xs,
            'y': ys,
            'vx': vxs,
            'vy': vys,
            'rho': rhos,
            'particle_id': i
        })
        
        all_data.append(particle_df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

def visualize_trajectories(df):
    plt.figure(figsize=(10, 10))
    
    # Draw the star
    circle = plt.Circle((0, 0), R, color='orange', alpha=0.3, label='Star (Dust Cloud)')
    plt.gca().add_patch(circle)
    
    # Plot trajectories
    for pid in df['particle_id'].unique():
        pdata = df[df['particle_id'] == pid]
        plt.plot(pdata['x'], pdata['y'], alpha=0.5, linewidth=1)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectories in Schwarzschild Spacetime')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectories.png')
    print("Saved trajectory plot to trajectories.png")

if __name__ == "__main__":
    # Generate data
    df = generate_data(num_particles=20, t_span=(0, 150))
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '../data/universe_data.csv')
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Visualize
    visualize_trajectories(df)
