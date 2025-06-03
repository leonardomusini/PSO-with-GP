# PSO-with-GP

This project combines **Particle Swarm Optimization (PSO)** with **Genetic Programming (GP)** to automatically evolve new velocity update rules for PSO.
Each GP individual represents a candidate update rule, expressed as a mathematical expression. The goal is to evolve rules that improve the performance of PSO across a suite of benchmark optimization problems.

## Structure of the repository

- **`PSO.py`**: Contains the implementation of both the Standard PSO algorithm and the Evolved PSO.

- **`GP.py`**: Implements the GP evolutionary algorithm using the DEAP framework.

- **`functions.py`**: Defines a set of benchmark objective functions used for training and testing.

- **`utils.py`**: Provides safe arithmetic operations used as primitives during GP tree evolution.

- **`analysis.ipynb`**: Used for running and visualizing the training and testing phases.