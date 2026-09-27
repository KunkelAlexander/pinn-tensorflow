# Solve the linear Schrödinger equation in 1D using a Physics Informed Neural Network

![PINN prediction of the real part of the wave function](figures/16_pred_surface_real.png)

This code heavily draws on the implementation of the PINN approach published by Jan Blechschmidt under https://github.com/janblechschmidt/PDEsByNNs/ (MIT license).

## Files

| File | Description |
|---|---|
| `schrodinger_pinn_pytorch.py` | **PyTorch implementation** — recommended starting point |
| `linear_schroedinger_1d.ipynb` | Original TensorFlow implementation (continuous time) |
| `linear_schroedinger_1d_discrete_time.ipynb` | Original TensorFlow implementation (discrete time) |

## Setup

### Create the conda environment

```bash
conda env create -f environment.yml
conda activate pinn_env
```

This installs a CPU-only PyTorch build. For GPU support, open `environment.yml`, remove the `cpuonly` line, and add `pytorch-cuda=12.1` (adjust the version to match your CUDA installation), then re-run the command above.

### Register the Jupyter kernel

```bash
python -m ipykernel install --user --name pinn_env --display-name "Python (pinn_env)"
```

### Run the PyTorch notebook

Open `schrodinger_pinn_pytorch.py` in VS Code — the Python extension treats `# %%` cell markers as a Jupyter notebook. Select the `pinn_env` kernel and run cells interactively, or launch JupyterLab:

```bash
jupyter lab schrodinger_pinn_pytorch.py
```

## Explanation
In the following, we will solve the dimensionless 1D Schrödinger equation with inhomogeneous Dirichlet boundary conditions using a Physics Informed Neural Network (PINN).
This boundary values problem can be stated as

$$i \hbar \partial_t \psi(x, t) = \left(-\frac{\hbar^2}{2m}\partial_x^2 + V(x, t)\right) \psi(x, t)$$
with
$$\psi(x, t = t_0) \equiv \psi_0(x) \qquad \psi(x=x_l, t) \equiv \psi_L(t) \qquad \psi(x=x_r, t) \equiv \psi_R(t)$$
for $t \in [t_0, t_1]$ and $x \in [x_l, x_r]$ and $\psi \in C_2([t_0, t_1] \times [x_l, x_r], \mathbb{C})$.

In the first approach considered here - the continuous time approach - we approximate $\psi(x, t)$ via a neural network $\psi_{\theta}(x, t)$ that takes two input parameters and outputs two ouput parameters - the real and imaginary part of the wave function. Note that we approximate the wave function in both spatial and temporal dimensions. In order for the NN to satisfy the boundary value problem, the residual of the PDE

$$r_{\theta}(x, t) \equiv \left(i \hbar \partial_t + \frac{\hbar^2}{2m} \partial_x^2 - V(x, t)\right)\psi_{\theta}(x, t)$$
is included in the loss functional. In total, the loss functional then contains three terms:
 - The mean squared residual
 - The mean squared misfit w.r.t. initial conditions
 - The mean squared misfit w.r.t. boundary conditions

It is minimised over a number of collocation points that are randomly sampled from $[t_0, t_1] \times [x_l, x_r]$.
