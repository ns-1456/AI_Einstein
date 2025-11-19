import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load Data
data_path = os.path.join(os.path.dirname(__file__), '../data/curvature_data.csv')
df = pd.read_csv(data_path)

X = df[['rho']].values
y = df['G_00'].values

print(f"Loaded {len(df)} samples.")
print(f"Rho range: {X.min()} to {X.max()}")
print(f"G_00 range: {y.min()} to {y.max()}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.xlabel('Mass Density (rho)')
plt.ylabel('Einstein Curvature (G_00)')
plt.title('Curvature vs Matter')
plt.grid(True)

try:
    from pysr import PySRRegressor
    print("PySR detected. Starting Symbolic Regression...")
    
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", "neg"],
        equation_file=os.path.join(os.path.dirname(__file__), 'hall_of_fame.csv')
    )
    
    model.fit(X, y)
    
    print("\nDiscovered Equations:")
    print(model.equations_)
    
    best_eq = model.sympy()
    print(f"\nBest Equation: G_00 = {best_eq}")
    
    # Plot prediction
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color='red', label='PySR Model')
    
except ImportError:
    print("PySR not installed. Falling back to Linear Regression.")
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    k = reg.coef_[0]
    c = reg.intercept_
    
    print(f"\nLinear Regression Result:")
    print(f"G_00 = {k:.4f} * rho + {c:.4f}")
    
    y_pred = reg.predict(X)
    plt.plot(X, y_pred, color='red', label=f'Linear Fit (k={k:.2f})')

plt.legend()
plt.savefig('discovery_plot.png')
print("Saved discovery_plot.png")
