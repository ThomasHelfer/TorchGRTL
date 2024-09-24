![Unittest](https://github.com/ThomasHelfer/TorchGRTL/actions/workflows/actions.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


# TorchGRTL - Numerical General Relativity in PyTorch
<p align="center">
    <img src="https://github.com/ThomasHelfer/TorchGRTL/blob/main/img/logo_cropped.png" alt="no alignment" width="25%" height="auto"/>
</p>



## Overview
TorchGRTL is a Python library for the application of deep learning to Numerical Relativity. It is based on the [GRTL codebase](https://example.com](https://github.com/GRTLCollaboration/GRChombo)) (formerly known as GRChombo) codebase. 

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

### Training models 

   To first learn a model please download the training dataset from <https://huggingface.co/datasets/thelfer/BinaryBlackHole> and adapt the path in yaml [<configs/factor_2.yaml>](https://github.com/ThomasHelfer/TorchGRTL/blob/main/configs/factor_2.yaml).

   ```yaml
factor: 2
# Data Source
filenamesX: "<ADAPT TO YOUR LOCAL PATH>/outputXdata_level{res_level}_step*.dat"
filenames_check: "/home/thelfer1/scr4_tedwar42/thelfer1/high_end_data_4/outputXdata_level{res_level}_step*.dat"
# Restarting
restart: False
   ```

  To run the code, simply run
   ```bash
   python learn_error.py configs/factor_2.yaml
   ```

### Evaluate models metrics 

  Copy your model fold in the /models folder and run 

   ```bash
   python evaluate_models.py 
   ```

  the corresponding metrics in the outputfile 'metrics_results.csv' can be visualised in notebook.ipynb


### Computing Constraints

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
