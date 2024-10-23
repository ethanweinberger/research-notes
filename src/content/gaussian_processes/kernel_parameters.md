---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{math}

\newcommand\x{\mathbf{x}}
\newcommand\k{\mathbf{k}}
\newcommand\f{\mathbf{f}}
\newcommand\y{\mathbf{y}}
\newcommand\v{\mathbf{v}}
\newcommand\0{\mathbf{0}}
```

# Gaussian Processes 3: Choosing the kernel parameters

In the previous section we applied GP regression with a squared exponential kernel

$$k(\x_i, \x_j) = \sigma^2_2 \exp \left(-\frac{1}{2\ell^2} (\x_i - \x_j)^{T}(\x_i - \x_j)\right),$$

with $\sigma$ and $\ell$ both fixed at 1.0. In practice, we typically won't know how to choose our kernel parameters _a priori_. This is a problem, as the choice of these parameters can have a major impact on our regression results. We illustrate this impact using our example from the previous section. First we generate data from the sine function with some added noise:

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    align: center
---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Noise free training data
np.random.seed(42)
noise = 0.1
n_train_points = 8
n_test_points = 100

domain = (-6, 6)
x_test = np.linspace(domain[0], domain[1], 100).reshape(-1, 1)
y_test = np.sin(x_test) 

x_train = np.random.uniform(low=domain[0] + 2, high=domain[1] - 2, size=(n_train_points, 1))
y_train = np.sin(x_train) + noise*np.random.normal(size=(n_train_points, 1))

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(x_test, y_test, '--', label="True function",)
ax.plot(x_train, y_train, 'ko', linewidth=2, label='Observed data')
ax.legend(loc='upper right')

sns.despine()
```

Now we fit GP regression to this data using varying values of $\sigma$ and $\ell$.

```{code-cell} ipython3

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def gp_posterior(sigma_f, l, noise):
    K = kernel(x_train, x_train, l, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K, upper=False)
    beta = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, beta)

    K_s = kernel(x_train, x_test, l, sigma_f)
    K_ss = kernel(x_test, x_test, l, sigma_f)

    v = np.linalg.solve(L, K_s)

    mu_s = K_s.T.dot(alpha).reshape(-1)
    diag_cov = np.diag(K_ss) - np.sum(v*v, axis=0)
    return mu_s, diag_cov

sigma_vals = (0.5, 1.0, 3.0)
ell_vals = (0.5, 1.0, 3.0)

fig, axes = plt.subplots(3, 3, figsize=(12, 7))

for i, sigma in enumerate(sigma_vals):
    for j, ell in enumerate(ell_vals):
        mu_s, diag_cov = gp_posterior(sigma, ell, noise)
        uncertainty = 1.96 * np.sqrt(diag_cov)

        ax = axes[i][j]
        ax.plot(x_test, y_test, '--', label='True function')
        ax.plot(x_test, mu_s, label='Posterior mean')
        ax.fill_between(
            x_test.reshape(-1),
            (mu_s + uncertainty).reshape(-1),
            (mu_s - uncertainty).reshape(-1),
            alpha=0.1,
            color='grey'
        )
        ax.plot(x_train, y_train, 'rx', label='Observed data')
        ax.set_ylim(-3, 3)
        ax.set_title(rf"$\sigma$ = {sigma}, $\ell$ = {ell}")
        sns.despine()

plt.subplots_adjust(hspace=0.5)
```

Clearly, the choice of hyperparameters has a major impact on our resulting model fit. Thus, we need some principled way for choosing a good set of hyperparameters.

## Optimizing the likelihood

For an arbitrary kernel $k$, let $\theta$ denote the set of the kernel's corresponding hyperparameters. In the case of the squared exponential kernel, we have $\theta = \{\sigma, \ell\}$.

A natural criterion for choosing $\theta$ is to maximize the marginal likelihood

```{math}
:label: likelihood_integral
p(\y \mid X, \theta) = \int p(\y \mid \f)p(\f \mid X, \theta)d\f
```

From our GP prior, we have

```{math}
p(\f \mid X, \theta) = \mathcal{N}(\0, K),
```

and from our definition of $\y$ we have

```{math}
p(\y \mid f) = \mathcal{N}(\f, \sigma_{\y}^2).
```

As all the terms in Equation {eq}`likelihood_integral` are Gaussian, the integral can be evaluated analytically and resulting (log) marginal likelihood is

```{math}
\log p(\y \mid X, \theta) = -\frac{1}{2}\y^{T}(K + \sigma_{\y}^{2}I)^{-1}\y - \frac{1}{2}\log |K + \sigma_{\y}^{2}I| - \frac{n}{2}\log 2\pi
```

Thus, our optimal set of parameters is 

```{math}
\theta^* = \arg\max_{\theta} \log p(\y \mid X, \theta)  = \arg\max_{\theta} \left(-\frac{1}{2}\y^{T}(K + \sigma_{\y}^{2}I)^{-1}\y - \frac{1}{2}\log |K + \sigma_{\y}^{2}I| - \frac{n}{2}\log 2\pi\right)
```

Equivalently, we can find $\theta^*$ by _minimizing_ the _negative_ log-likelihood

```{math}
\theta^* = \arg\min_{\theta} -\log p(\y \mid X, \theta)  = \arg\min_{\theta} \left(\frac{1}{2}\y^{T}(K + \sigma_{\y}^{2}I)^{-1}\y + \frac{1}{2}\log |K + \sigma_{\y}^{2}I| + \frac{n}{2}\log 2\pi\right)
```

which allows us to work with off-the-shelf optimization tools that expect minimization problems.

## Implementation

We now proceed to implement the above procedure. Note that when computing the negative log-likelihood we avoid numerically unstable matrix inversions using the procedure described in the previous chapter.

```{code-cell} ipython3
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from numpy.linalg import cholesky, det

def nll_stable(theta):
    # Numerically more stable implementation of Eq. (11) as described
    # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
    # 2.2, Algorithm 2.1.

    K = kernel(x_train, x_train, l=theta[0], sigma_f=theta[1]) + noise * np.eye(len(x_train))
    L = cholesky(K)

    beta = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, beta)
    
    return np.sum(np.log(np.diagonal(L))) + \
           0.5 * y_train.ravel().dot(alpha) + \
           0.5 * len(x_train) * np.log(2*np.pi)

# Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
# We should actually run the minimization several times with different
# initializations to avoid local minima but this is skipped here for
# simplicity.
res = minimize(nll_stable, [1, 1], 
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')

# Store the optimization results in global variables so that we can
# compare it later with the results from other implementations.
l_opt, sigma_f_opt = res.x

def gp_posterior(sigma_f, l, noise):
    K = kernel(x_train, x_train, l, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K, upper=False)
    beta = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, beta)

    K_s = kernel(x_train, x_test, l, sigma_f)
    K_ss = kernel(x_test, x_test, l, sigma_f)

    v = np.linalg.solve(L, K_s)

    mu_s = K_s.T.dot(alpha).reshape(-1)
    diag_cov = np.diag(K_ss) - np.sum(v*v, axis=0)
    return mu_s, diag_cov

mu_s, diag_cov = gp_posterior(sigma_f_opt, l_opt, noise)

fig, ax = plt.subplots()
uncertainty = 1.96 * np.sqrt(diag_cov)

ax.plot(x_test, y_test, '--', label='True function')
ax.plot(x_test, mu_s, label='Posterior mean')
ax.fill_between(
    x_test.reshape(-1),
    (mu_s + uncertainty).reshape(-1),
    (mu_s - uncertainty).reshape(-1),
    alpha=0.1,
    color='grey'
)
ax.plot(x_train, y_train, 'rx', label='Observed data')
ax.legend(loc='upper right')
sns.despine()
```

From the above plot, it appears that our objective function indeed leads to reasonable hyperparameter values.