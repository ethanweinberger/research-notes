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
```

# Gaussian Processes 2: Gaussian process regression

In the previous section we introduced stochsatic processes and defined the Gaussian process. In this section we'll demonstrate how Gaussian processes will allow us to solve regression tasks while naturally incorporating estimates of uncertainty.


## The noise-free case

Suppose we're given values of our function $f(\mathbf{x}_i) \in \mathbb{R}$ evaluated  at $n$ input points $\mathbf{x}_i \in \mathbb{R}^{d}$. We'll sometimes refer to this observed data as our _training data_. To simplify our notation, we'll use the matrix $X \in \mathbb{R}^{n \times d}$ to denote matrix of function input values for our training data, and we'll correspondingly use $\mathbf{f} \in \mathbb{R}^{n}$ to represent our training data's function outputs. Now let's assume that $f$ is drawn from a Gaussian process, i.e.,

$$f \sim \mathcal{GP}(\mathbf{0}, k(\mathbf{x}, \mathbf{x}'))$$

with the kernel $k$ chosen to reflect some prior belief about how the outputs of our function vary with respect to the input values. Our task now to predict the values of $f$ at a collection of test points $X_*$ for which we don't observe the corresponding outputs $\mathbf{f}_*$. Based on our Gaussian process asssumption, we have

$$
\begin{eqnarray}
\left[
    \begin{array}{l}
    \ \mathbf{f} \\ 
    \ \mathbf{f_*}
    \end{array}
 \right]
\end{eqnarray} \sim \mathcal{N}\left(\boldsymbol{0}, \begin{bmatrix} K(X, X) & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*)\end{bmatrix}\right)$$ 

where $K(X,X)$ denotes the $n \times n$ matrix where the $(i, j)$'th entry corresponds to our kernel function evaluated at the training
inputs $\mathbf{x}_i$ and $\mathbf{x}_j$ (i.e., $k(\mathbf{x}_i, \mathbf{x}_j)$). We define $K(X, X_*)$, $K(X_*, X)$, and $K(X_*, X_*)$ 
analogously for pairs of training and/or test inputs. From the conditioning property of Gaussians, we then immediately obtain:

```{math}
:label: gp_posterior
\mathbf{f_*} \mid X_*, X, \mathbf{f} \sim \mathcal{N}(K(X_*, X)K(X, X)^{-1}\mathbf{f}, K(X_*, X_*) - K(X_*, X)K(X, X)^{-1}K(X, X_*))
```

We can then sample function values $\mathbf{f_*}$ corresponding to our test inputs $X_*$ by sampling from the above distribution, and this procedure is known as _Gaussian process regression_.

The notation in Equation {eq}`gp_posterior` can quickly get cumbersome. To simplify things a bit, define $K = K(X, X)$, $K_{*} = K(X, X_*)$, and $K_{**} = K(X_*, X_*)$. We may then rewrite Equation {eq}`gp_posterior` as

```{math}
:label: gp_posterior_cleaner
\mathbf{f_*} \mid X_*, X, \mathbf{f} \sim \mathcal{N}(K_{*}^{T}K^{-1}\mathbf{f}, K_{**} - K_{*}^{T}K^{-1}K_{*}).
```

To further simplify things, when making predictions for a single test input $\x_*$ we define $\k_*$ as the vector of covariances between the test point and training points. For predictions at a single test input we then can rewrite Equation {eq}`gp_posterior_cleaner` as

```{math}
:label: gp_posterior_single_point
\mathbf{f_*} \mid \x_*, X, \mathbf{f} \sim \mathcal{N}(\k_*^{T}K^{-1}\mathbf{f}, k(\x_*, \x_*) - \k_*^{T}K^{-1}\k_{*}).
```

## Implementation

We'll make this procedure more concrete by illustrating it on a toy problem. Suppose we have
data generated from an underlying sine function

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
n_train_points = 8
n_test_points = 100

domain = (-6, 6)
x_test = np.linspace(domain[0], domain[1], 100).reshape(-1, 1)
y_test = np.sin(x_test)

x_train = np.random.uniform(low=domain[0] + 2, high=domain[1] - 2, size=(n_train_points, 1))
y_train = np.sin(x_train)

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(x_test, y_test, '--', label="True function",)
ax.plot(x_train, y_train, 'ko', linewidth=2, label='Observed data')
ax.legend(loc='upper right')

sns.despine()
```

where the black dots are our observed data points and the dashed blue curve is our full underlying function. We'll now apply GP regression to obtain predictions for values at unseen inputs.

For now, for our covariance function we'll use the squared exponential kernel

$$ k(\x_i, \x_j) = \sigma^2 \exp \left(-\frac{1}{2\ell^2} (\x_i - \x_j)^{T}(\x_i - \x_j)\right) $$

Here the length parameter $\ell$ controls the smoothness of the function and $\sigma$ controls the vertical variation. For this example we'll fix $\sigma$ and $\ell$ at 1.0; we discuss how to choose these parameters later.

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
```

We can then solve for our posterior's mean and covariance parameters


```{code-cell} ipython3
from jax.numpy.linalg import inv
l = 1.0
sigma_f = 1.0
jitter = 1e-5 # To help with numerical stability issues

K = kernel(x_train, x_train, l, sigma_f) + jitter * np.eye(len(x_train))
K_s = kernel(x_train, x_test, l, sigma_f)
K_ss = kernel(x_test, x_test, l, sigma_f)
K_inv = inv(K)

# Equation (7)
mu_s = K_s.T.dot(K_inv).dot(y_train).reshape(-1)

# Equation (8)
cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
```

And we then plot the results

```{code-cell} ipython3
---
mystnb:
  image:
    align: center
---
fig, ax = plt.subplots()
uncertainty = 1.96 * np.sqrt(np.diag(cov_s))

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
plt.show()
```

where the solid orange line denotes our posterior mean (i.e., the center of our predicted function values at unknown points) and the shaded area represents how uncertain we are about predictions at individual inputs (as captured by two times the posterior standard deviation). We find that our GP regression procedure fits the underlying function reasonably well for areas of the input space where we have observed data. Moreover, for inputs where our model does not have nearby observed data points and thus fails to make accurate predictions, the posterior distribution has a large degree of uncertainty.

## The noisy case

In the previous section we assumed that our training dataset contains the true function values $f(\mathbf{x}_i)$ for each training data input $\mathbf{x}_i$. In most realistic modeling scenarios, we won't be so lucky to have the true function values. Instead, we might have noisy outputs

$$ y_i = f(\mathbf{x}_i) + \varepsilon $$

where our noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. With this assumption, the covariance between any two evaluations of our $f$ at points $\mathbf{x}_i$ and $\mathbf{x}_j$ becomes

$$ cov(\mathbf{x}_i, \mathbf{x}_j) = k(\mathbf{x}_i, \mathbf{x}_j) + \delta_{ij}\sigma^2 $$

where $\delta_{ij}$ is one if $i = j$ and zero otherwise; this reflects our assumption that the noise $\varepsilon$ is independent from the value of our function inputs. Letting $\mathbf{y} \in \mathbb{R}^{n}$ denote our noisy outputs $\{y_i\}$ collected into a single vector, we can equivalently write

$$ cov(\mathbf{y}) = K + \sigma^2I $$

With this additional noise term, our joint distribution for training and test point outputs then becomes

```{math}
:label: joint_noisy
\begin{eqnarray}
\left[
    \begin{array}{l}
    \ \mathbf{y} \\ 
    \ \mathbf{f_*}
    \end{array}
 \right]
\end{eqnarray} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K + \sigma^2I & K_{*} \\ K_{*}^{T} & K_{**} \end{bmatrix}\right)
```

Just as before, we can apply the conditioning property of multivariate normal distributions to Equation {eq}`joint_noisy` to obtain

$$ \mathbf{f_*} \mid X_*, X, \mathbf{y} \sim \mathcal{N}(\bar{\mathbf{\f}}_*, \mathbf{\Sigma}_*),$$

where 

```{math}
:label: posterior_mean_noisy
\bar{\mathbf{f}}_* = K_{*}^{T}(K + \sigma^2I)^{-1}\mathbf{y}
```

```{math}
:label: posterior_variance_noisy
\mathbf{\Sigma}_* = K_{**} - K_{*}^{T}(K + \sigma^2I)^{-1}K_{*}.
```

For a single test point $\x_*$ we can simplify Equation {eq}`posterior_mean_noisy` as

```{math}
:label: posterior_mean_noisy_single_point
\bar{f}_* = \k_{*}^{T}(K + \sigma^2I)^{-1}\mathbf{y}
```

and {eq}`posterior_variance_noisy` as

```{math}
:label: posterior_variance_noisy_single_point
\sigma_* = \k(\x_*, \x_*) - \k_{*}^{T}(K + \sigma^2I)^{-1}\k_{*}.
```


## Implementation (noisy case)

Using our previous example, we generate data with noise

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    align: center
---
noise = 0.1
domain = (-6, 6)
x_test = np.linspace(domain[0], domain[1], 100).reshape(-1, 1)
y_test = np.sin(x_test)

y_train = np.sin(x_train) + 0.5*np.random.normal(size=(n_train_points, 1))

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(x_test, y_test, '--', label="True function",)
ax.plot(x_train, y_train, 'ko', linewidth=2, label='Observed data')
ax.legend(loc='upper right')

sns.despine()
```

Applying Equations {eq}`posterior_mean_noisy` and {eq}`posterior_variance_noisy` we have

```{code-cell} ipython3
---
mystnb:
  image:
    align: center
---
l = 1.0
sigma_f = 1.0

K = kernel(x_train, x_train, l, sigma_f) + noise * np.eye(len(x_train))
K_s = kernel(x_train, x_test, l, sigma_f)
K_ss = kernel(x_test, x_test, l, sigma_f)
K_inv = inv(K)

mu_s = K_s.T.dot(K_inv).dot(y_train).reshape(-1)
cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
```

which we then visualize as

```{code-cell} ipython3
---
mystnb:
  image:
    align: center
---
fig, ax = plt.subplots()
uncertainty = 1.96 * np.sqrt(np.diag(cov_s))

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
plt.show()
```

Note how our posterior mean no longer passes through the original data points, as we no longer assume that they represent true function values.

## Numerical stability issues

Our previous implementations have been "correct" in that they match our derived expressions for the posterior mean and covariance. However, matrix inversion can be numerically unstable, rendering the $K^{-1}$ term (in the noise-free case) and the $(K + \sigma^2I)^{-1}$ term (in the noisy case) potentially problematic.

Fortunately, we can exploit the structure of our matrix to achieve a more stable implementation. For any positive-define matrix $A$ (e.g. our kernel matrix $K$) we can decompose $A$ as

```{math}
:label: cholesky
A = LL^{T}
```

This is known as the _Cholesky_ decomposition. Now, starting from the definition of the matrix inverse we have

```{math}
:label: inverse
AA^{-1} = I.
```

Substituting $A$ in Equation {eq}`inverse` using the expression from Equation {eq}`cholesky` we have

```{math}
LL^{T}A^{-1} = I.
```

Now we can right-multiply both sides of this equation by an arbitrary vector $\y$ to give

```{math}
:label: y
LL^{T}A^{-1}\y = \y.
```

Which we can write as 

```{math}
:label: lty
LL^{T}A^{-1}\y = LT\y.
```

for some $T$. This then implies 

```{math}
:label: cholesky_penultimate
L^{T}\underbrace{A^{-1}\y}_{\gamma} = \underbrace{T\y}_{\omega}.
```

Now, _why_ did we go through all this extra effort? Note that the terms $A^{-1}\y$ and $T\y$ are both vectors. Thus, Equation {eq}`cholesky_penultimate` can be rewritten as 

```{math}
:label: cholesky_compact
L^{T}\alpha = \beta.
```

Importantly, if we can find the value of $\beta$, solving the above linear equation for $\alpha$ is numerically stable, and allows us to compute $A^{-1}\y$ without inverting $A$ directly. To solve for $\beta$, note that from Equations {eq}`y` and {eq}`lty` we have

```{math}
L\underbrace{Ty}_{\beta} = y.
```

From this expression we write

```{math}
\beta = L\ \backslash\ \y
```

where the notation $L\ \backslash\ \y$ denotes the vector that results in $\y$ when multipled by $L$. Substituting this result into {eq}`cholesky_compact` we then have

```{math}
L^{T} \alpha = L\ \backslash\ \y.
```

Therefore, 

```{math}
\alpha = L^{T}\ \backslash\ (L\ \backslash\ \y) = A^{-1}\y.
```

and 

```{math}
\bar{f}_{*} = \k_{*}^{T}\alpha.
```

Thus, we can compute our posterior mean by solving two systems of equations rather than directly inverting any matrices. Now we proceed similarly for the posterior variance.

We have 

```{math}
\sigma^2_{*} = k(\x_*, \x_*) - \k_*^TA^{-1}\k_*,
```

with a problematic matrix inverse in the $\k_*^TA^{-1}\k_*$ term. Proceeding via Cholesky, we have

```{math}
\k_*^TA^{-1}\k_* = \k_*^T(LL^{T})^{-1}\k_* = \k_*^{T}(L^{T})^{-1}L^{-1}\k_* = \v^{T}\v,
```

where $\v = L^{-1}\k_*$. Notably, we can solve for $\v$ without directly computing the inverse as we can write $L\v = \k_*$ so $\v = L\ \backslash\ \k_*$. We then have

```{math}
\sigma^2_{*} = k(\x_*, \x_*) - \v^{T}\v,
```

We reimplement Gaussian process regression for the noisy case using this idea below:

```{code-cell} ipython3
---
mystnb:
  image:
    align: center
---
K = kernel(x_train, x_train, l, sigma_f) + noise * np.eye(len(x_train))
L = np.linalg.cholesky(K, upper=False)
beta = np.linalg.solve(L, y_train)
alpha = np.linalg.solve(L.T, beta)

K_s = kernel(x_train, x_test, l, sigma_f)
K_ss = kernel(x_test, x_test, l, sigma_f)

v = np.linalg.solve(L, K_s)

mu_s = K_s.T.dot(alpha).reshape(-1)
diag_cov = np.diag(K_ss) - np.sum(v*v, axis=0)

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