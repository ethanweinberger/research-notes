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
\newcommand\y{\mathbf{y}}
\newcommand\f{\mathbf{f}}
\newcommand\X{\mathbf{X}}
\newcommand\x{\mathbf{x}}
\newcommand\k{\mathbf{k}}
\newcommand\K{\mathbf{K}}
\newcommand\u{\mathbf{u}}
\newcommand\0{\mathbf{0}}
\newcommand\Q{\mathbf{Q}}
```

# Gaussian Processes 4: Sparse GPs with pseudo-data

In the previous section we discussed how to perform Gaussian process regression to estimate the values of a function $f$. Our solution relied on the convenient properties of the multivariate Guassian distribution, which allowed us to compute posterior distributions of unknown function values in closed form.

Unfortunately, while it's straightforward to write down the equations for the posterior distribution, actually _computing_ it is non-trivial. Specifically, the posterior distribution requires inverting the covariance matrix $K$, which requires $\mathcal{O}(N^3)$ operations; thus as the size of our training data increases, this operation quickly becomes infeasible.

To work around this issue, a number of works have proposed so-called _sparse_ Gaussian process methods. At a high level, these methods apply different techniques to approximate the information captured by the full covariance matrix $K$ with a cheaper set of operations. In this chapter we'll review the method of {cite:t}`snelson2005sparse`, sometimes referred to as the fully independent training conditional (FITC) approximation.

## An example

Consider the following dataset:

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

def func(x):
    """Latent function."""
    return 1.0 * np.sin(x * 3 * np.pi) + 0.3 * np.cos(x * 9 * np.pi) + 0.5 * np.sin(x * 7 * np.pi)


# Number of training examples
n = 1000

# Number of inducing variables
m = 30

# Noise
sigma_y = 0.2

# Noisy training data
X = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
y = func(X) + sigma_y * np.random.normal(size=(n, 1))

# Test data
X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
f_true = func(X_test)

# Inducing inputs
X_m = np.linspace(-0.4, 0.4, m).reshape(-1, 1)

fig, ax = plt.subplots()
ax.scatter(X, y, label='Training examples', marker='x', color='blue', alpha=0.1)
ax.plot(X_test, f_true, label='Latent function', c='k', lw=0.5)
ax.set_title('Dataset')
ax.legend(loc='upper right')
sns.despine()
```

Intuitively, in order to fit this function we probably don't need to consider _all_ of the observed data when fitting our model. Instead, if we chose a smaller subset of points that reflect the general behavior of our function, we could probably achieve a similar fit as compared to using the whole dataset. 

Based on this idea, we'll define an alternative generative process as follows. Suppose that, in addition to our observed dataset with inputs $\X$ and (noisy) outputs $\y$, we also have a _pseudo dataset_ with inputs $\bar{\X}$ and outputs $\u$. As our pseudo data points are not "real" observations, we assume that they represent true function evaluations and are noise-free. Given our pseudo-data, we assume that each data point has the following likelihood

```{math}
:label: pseudo_likelihood
p(y_n \mid \x_n, \bar{\X}, \u) = \mathcal{N}(y \mid \k_{\x_n}^{T}\K_{M}^{-1}\u,\ \underbrace{K_{\x_n\x_n} - \k_{\x_n}^{T}\K_{M}^{-1}\k_{x_n}}_{\lambda_n} + \sigma^2)
```

We note two facts about the likelihood in Equation {eq}`pseudo_likelihood`. First, our likelihood strongly resembles that of standard Gaussian process regression, with the original inverse kernel matrix now replaced by our pseudo-data's kernel matrix. Second, as our pseudo-data are assumed to be noise free, we only see $\sigma^2$ appear as an additional summand in our likelihood's variance term (and not in the kernel matrix inverses).

Further assuming that our data are generated independently given the pseudo dataset, our full dataset likelihood is then

```{math}
:label: pseudo_likelihood_full
p(\y \mid \X, \bar{\X}, \u) = \prod_{n}p(y_n \mid \x_n, \bar{\X}, \u) = \mathcal{N}(\y \mid \K_{NM}\K_{M}^{-1}\u,\ \mathbf{\Lambda} + \sigma^2I).
```

where $\mathbf{\Lambda} = \text{diag}(\{\lambda_n\})$. Now that we've specified our model, we must 


We could then learn $\u$ and $\bar{\X}$ that maximize Equation {eq}`pseudo_likelihood_full` and make predictions for new data points using Equation {eq}`pseudo_likelihood`. However, optimizing over both the $\u$ and $\bar{\X}$ terms at the same time can be tricky. 

To alleviate this issue, we can instead derive predictive distributions and a training objective that don't depend on $\u$ by marginalizing this term out. First let's consider the posterior predictive distribution:

```{math}
:label: posterior_predictive
\begin{align}
p(y_* \mid \x_*, \y, \X, \bar{\X}) &= \int p(y_* \mid \x_*, \y, \X, \bar{\X}, \u)p(\u \mid \y, \X, \bar{\X})d\u \\
&= \int \underbrace{p(y_* \mid \x_*, \bar{\X}, \u)}_{\text{Likelihood}}\underbrace{p(\u \mid \y, \X, \bar{\X})}_{\text{Posterior}}d\u
\end{align}
```

From Equation {eq}`pseudo_likelihood`, we know that our Likelihood term is Gaussian. For the Posterior term we have:

```{math}
:label: posterior
p(\u \mid \y, \X, \bar{\X}) = \frac{p(\y \mid \X, \bar{\X}, \u)p(\u \mid \X, \bar{\X})}{p(\y \mid \X, \bar{\X})} = \frac{p(\y \mid \X, \bar{\X}, \u)p(\u \mid \bar{\X})}{p(\y \mid \X, \bar{\X})}.
```

To compute this we'll first need to specify the prior $p(\u \mid \bar{\X})$. Using the intuition that our pseudo data should be have similarly to real data, we'll set $p(\u) = \mathcal{N}(\u \mid \0, \K_{M})$. Beyond satisfying our intuition, this prior also has the benefit of ensuring that all the terms in the numberator of Equation {eq}`posterior` are Gaussian. From the properties of Gaussians, we can then infer

```{math}
p(\u \mid \y, \X, \bar{\X}) = \mathcal{N}(\u \mid \K_{M}\Q_{M}^{-1}\K_{MN}(\mathbf{\Lambda} + \sigma^2I)^{-1}\y, \K_{M}\Q_{M}^{-1}\K_{M}),
```

Plugging this into Equation {eq}`posterior_predictive`, we then have (again from the properties of Gaussians)

```{math}
p(y_* \mid \x_*, \y, \X, \bar{\X}) = \mathcal{N}(y_* \mid \k_{*}^{T}\Q_{M}^{-1}\K_{MN}(\mathbf{\Lambda} + \sigma^2I)^{-1}\y, K_{**} - \k_{*}^{T}(\K_{M}^{-1} - \Q_{M}^{-1})\k_{*} + \sigma^2).
```

Thus, we have a principled way for making predictions with test points without needing to rely on the values of $\u$. Next, we need an objective to optimize for $\bar{\X}$. To do so, we'll consider the marginal likelihood

```{math}
p(\y \mid \X, \bar{\X}, \u) = \int p(\y \mid \X, \bar{\X}, \u)p(\u \mid \bar{X})d\u
```

From the properties of Gaussians, we have

```{math}
p(\y \mid \X, \bar{\X}, \u) = \mathcal{N}(\y \mid \0, \K_{NM}\K_{M}^{-1}\K_{MN} + \mathbf{\Lambda} + \sigma^2I),
```

which we can maximize via gradient ascent.

## Implementation



---
```{bibliography}
```