# Gaussian Processes 2: GP Regression

In the previous section we introduced the Gaussian process and reviewed some useful properties of Gaussian
random variables. In this section we'll demonstrate how these properties will allow us to solve regression
tasks while naturally incorporating estimates of uncertainty.

---

Suppose we're given a training dataset consisting of $n$ input points $\{x_i\}_{i=1}^{n}$ as well as
the corresponding outputs from a function $f(x_i)$. Based on some prior knowledge of $f$, we'll assume that $f$
follows a GP prior

$$f \sim \mathcal{GP}(0, k(x, x'))$$

with the kernel $k$ chosen to reflect our knowledge of the world. Our task now to predict the values of $f$ at a collection
of test points $X_*$. To do so, we'll exploit the fact that our function evaluations form a multivariate normal distribution, with

$$ \begin{bmatrix} f \\ f_* \end{bmatrix} \sim \mathcal{N}\left(\boldsymbol{0}, \begin{bmatrix} K(X, X) & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*)\end{bmatrix}\right)$$ 

With this fact and 

## Optimizing the kernel parameters