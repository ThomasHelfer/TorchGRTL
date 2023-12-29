from GeneralRelativity.Utils import get_box_format, TensorDict, cut_ghosts, keys, keys_all
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2, FOR3, FOR4
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from GeneralRelativity.TensorAlgebra import compute_christoffel, compute_trace, compute_christoffel_fast, raise_all
from GeneralRelativity.CCZ4Geometry import compute_ricci
from GeneralRelativity.Constraints import constraint_equations
import matplotlib.pyplot as plt
import torch
import numpy as np

# Load data from files
filenamesX = "tests/TestData/Xdata_level0_step*"
num_varsX = 104
dataX = get_box_format(filenamesX, num_varsX)

# Set differential value for computation
oneoverdx = 64.0 / 4.0

# Process data by cutting ghosts and creating tensor dictionaries
dataXcut = cut_ghosts(dataX)
vars = TensorDict(dataXcut, keys_all)
d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
d2 = TensorDict(diff2(dataX, oneoverdx**2), keys_all)

# Compute the inverse metric tensor and Christoffel symbols
h_UU = torch.inverse(vars['h'])
chris = compute_christoffel(d1['h'], h_UU)
chrisref = compute_christoffel_fast(d1['h'], h_UU)

# Calculate constraints based on the variables and their derivatives
out = constraint_equations(vars, d1, d2, h_UU, chris)

# Extract the data for plotting
ham_data = out['Ham'][0, 6, ...]

# Create a meshgrid for the x and y axes
nx, ny = ham_data.shape
x = np.linspace(0, nx / oneoverdx, nx)
y = np.linspace(0, ny / oneoverdx, ny)
X, Y = np.meshgrid(x, y)

# Plotting
plt.figure(figsize=(8, 6))  # Set the figure size
plt.pcolormesh(X, Y, ham_data, cmap='viridis', shading='auto')
plt.colorbar(label='Hamiltonian Constraint')  # Adds a color bar with a label
plt.title('Hamiltonian Constraint Visualization')
plt.xlabel('X (M)')
plt.ylabel('Y (M)')
plt.savefig('ham.png')