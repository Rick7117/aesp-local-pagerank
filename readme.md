# Accelerated Evolving Set Processes for Local PageRank Computation

## Overview

This repository contains the implementation and experimental code for our paper "Accelerated Evolving Set Processes for Local PageRank Computation" submitted to NIPS 2025.

## Abstract

This work proposes a novel framework based on nested evolving set processes to accelerate Personalized PageRank (PPR) computation. At each stage of the process, we employ a localized inexact proximal point iteration to solve a simplified linear system. We show that the time complexity of such localized methods is upper bounded by min{Õ(R²/ε²), Õ(m)} to obtain an ε-approximation of the PPR vector, where m denotes the number of edges in the graph and R is a constant defined via nested evolving set processes.

Furthermore, the algorithms induced by our framework require solving only Õ(1/√α) such linear systems, where α is the damping factor. When 1/ε² ≪ m, this implies the existence of an algorithm that computes an ε-approximation of the PPR vector with an overall time complexity of Õ(R²/(√α·ε²)), independent of the underlying graph size. Our result resolves an open conjecture from existing literature.

## Paper

📄 **ArXiv Paper**: [https://arxiv.org/abs/2510.08010](https://arxiv.org/abs/2510.08010)

## Key Contributions

- Novel framework based on nested evolving set processes for PPR computation
- Theoretical analysis showing improved time complexity bounds
- Resolution of an open conjecture from existing literature
- Experimental validation on real-world graphs demonstrating significant convergence improvements

## Repository Structure

```
.
├── src/                 # Source code implementation
├── experiments/         # Experimental scripts and configurations
├── data/               # Dataset files and preprocessing scripts
├── results/            # Experimental results and analysis
└── README.md           # This file
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- NetworkX
- Matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/username/NIPS2025.git
cd NIPS2025
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.accelerated_ppr import AcceleratedPPR

# Load your graph
G = load_graph('path/to/graph')

# Initialize the accelerated PPR solver
solver = AcceleratedPPR(damping_factor=0.85, epsilon=1e-6)

# Compute PPR vector
ppr_vector = solver.compute_ppr(G, source_node)
```

### Running Experiments

```bash
# Run experiments on synthetic graphs
python experiments/run_synthetic.py

# Run experiments on real-world graphs
python experiments/run_realworld.py
```

## Experimental Results

Our experimental results on real-world graphs validate the efficiency of our methods, demonstrating significant convergence in the early stages compared to existing approaches.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{huang2024accelerated,
  title={Accelerated Evolving Set Processes for Local PageRank Computation},
  author={Huang, Binbin and others},
  journal={arXiv preprint arXiv:2510.08010},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Binbin Huang: [email]

## Acknowledgments

We thank the reviewers and colleagues for their valuable feedback and suggestions.