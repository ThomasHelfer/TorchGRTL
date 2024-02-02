![Unittest](https://github.com/ThomasHelfer/TorchGRTL/actions/workflows/actions.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


# TorchGRTL

## Overview
TorchGRTL is a Python library that provides a PyTorch-based implementation of key components of [GRTL codebase](https://example.com](https://github.com/GRTLCollaboration/GRChombo)) . It leverages the power of PyTorch to accelerate computations, making it suitable for machine learning applications in physics and simulation.

## Key Features:

Unleash the Power of PyTorch for General Relativity:

- Fast and seamless computations: Perform complex general relativity simulations with the efficiency and flexibility of PyTorch, a leading deep learning framework.
High-Performance Interpolation for Precision:

Accurate and efficient:
- Achieve high-order polynomial interpolation with exceptional performance using a convolutional network architecture, ensuring precise results for your simulations.
Seamless Integration with Machine Learning:

Bridging the gap: 
 - Effortlessly incorporate general relativity calculations into your machine learning workflows, unlocking a universe of possibilities for interdisciplinary research.

Confidence through testing: 
 - Trust in the codebase's robustness and accuracy, as it's rigorously developed and maintained using test-driven development practices.

## Installation

### Prerequisites
Before installing TorchGRTL, ensure you have the following prerequisites:
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the TorchGRTL repository:

   ```bash
   git clone https://github.com/ThomasHelfer/TorchGRTL.git
   cd TorchGRTL
   ```

2. Install the package:

   ```bash
   pip install .
   ```
3. (Optional) Set up pre-commit hooks for code formatting and linting:

   ```bash
   pre-commit install
   ```

## Usage

   The TorchGRTL library offers powerful tools to compute a variety of quantities essential in numerical relativity. Here are some examples of how you can use the library:

### Computing Christoffel Symbols

   You can compute the Christoffel symbols, which are crucial in the context of general relativity for defining the Levi-Civita connection and geodesic equations:

   ```python
   # Compute the Christoffel symbols using the standard method
   chris = compute_christoffel(d1['h'], h_UU)
   ```

   In these examples, d1['h'] refers to the first derivatives of the metric tensor, and h_UU is the inverse metric tensor.

   Calculating Hamiltonian and Momentum Constraints
   The library can also compute more complex quantities like the Hamiltonian and Momentum constraints, which are fundamental in ensuring the consistency of solutions in numerical relativity:

   ```python
   # Compute the Hamiltonian and Momentum constraints
   out = constraint_equations(vars, d1, d2, h_UU, chris)
   ```
   Here, vars contains various tensor fields, d1 and d2 are the first and second derivatives of these tensor fields, and chris is the computed Christoffel symbols.

### Self-Contained Example
   For a full, self-contained example that demonstrates the library's capabilities, refer to example.py in the repository. This example will guide you through a typical use case, showing how to leverage TorchGRTL for numerical relativity simulations and calculations.

### License

TorchGRTL is released under the MIT License. See LICENSE for more details.

### Contact

For questions or support, please contact Thomas Helfer at thomashelfer@live.de.
