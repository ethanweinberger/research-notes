# Gaussian Processes 2: GP Regression

In the previous section we introduced the Gaussian process and reviewed some useful properties of Gaussian
random variables. In this section we'll demonstrate how these properties will allow us to solve regression
tasks while naturally incorporating estimates of uncertainty.

---

Suppose we're given a training dataset consisting of $n$ input points $\mathbf{x}_i \in \mathbb{R}^{d}$ as well as
the corresponding outputs from a function $f(\mathbf{x}_i) \in \mathbb{R}$. To make things easier in terms of notation, we'll
sometimes denote our training inputs as a matrix $X \in \mathbb{R}^{n \times d}$, and our training outputs as $\mathbf{f} \in
\mathbb{R}^{n}$.

Now let's assume that $f$ is drawn from a GP prior

$$f \sim \mathcal{GP}(\mathbf{0}, k(\mathbf{x}, \mathbf{x}'))$$

with the kernel $k$ chosen to reflect some prior belief about how the outputs of our function vary with respect to the input values.
Our task now to predict the values of $f$ at a collection of test points $X_*$ for which we don't observe the outputs of our function 
$\mathbf{f}_*$. Based on our Gaussian process asssumption, we have

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
similarly for pairs of training and/or test inputs.

With this fact and the conditioning property of Gaussians, we then immediately obtain:

$$\mathbf{f_*} \mid X_*, X, \mathbf{f} \sim \mathcal{N}(K(X_*, X)K(X, X)^{-1}\mathbf{f}, K(X_*, X_*) - K(X_*, X)K(X, X)^{-1}K(X, X_*))$$

We can then sample function values $\mathbf{f_*}$ corresponding to our test inputs $X_*$ by sampling from the above distribution.

---
#### The noisy case

In the previous section we assumed that our training dataset contains the true function values $f(\mathbf{x}_i)$ for each input $\mathbf{x}_i$. In most realistic modeling scenarios, we won't be so lucky to have the true function values. Instead, we might have noisy outputs

$$ y_i = f(\mathbf{x}_i) + \varepsilon $$

where our noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. With this assumption, the covariance between any two evaluations of our $f$ at points $\mathbf{x}_p$ and $\mathbf{x}_q$ becomes

$$ cov(\mathbf{x}_p, \mathbf{x}_q) = k(\mathbf{x}_p, \mathbf{x}_q) + \delta_{pq}\sigma^2 $$

where $\delta_{pq}$ is one if $p = q$ and zero otherwise; this reflects our assumption that the noise $\varepsilon$ is independent
from the value of our function inputs. Letting $\mathbf{y} \in \mathbb{R}$ denote our noisy outputs $\{y_i\}$ collected into a single vector, we can equivalently write

$$ cov(\mathbf{y}) = K(X, X) + \sigma^2I $$

With this additional noise term, our joint distribution for training and test point outputs then becomes

$$
\begin{eqnarray}
\left[
    \begin{array}{l}
    \ \mathbf{y} \\ 
    \ \mathbf{f_*}
    \end{array}
 \right]
\end{eqnarray} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K(X, X) + \sigma^2I & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*)\end{bmatrix}\right)
$$

Just as before, we can apply the conditioning property of multivariate normal distributions to obtain

$$ \mathbf{f_*} \mid X_*, X, \mathbf{y} \sim \mathcal{N}(\mathbf{\mu}_*, \mathbf{\Sigma}_*) $$

where 

$$ \mathbf{\mu}_* = K(X_*, X)(K(X, X) + \sigma^2I)^{-1}\mathbf{y} $$

and

$$ \mathbf{\Sigma}_* = K(X_*, X_*) - K(X_*, X)(K(X, X) + \sigma^2I)^{-1}K(X, X_*) $$

#### Choosing the kernel parameters

So far we've assumed a fixed kernel function $k(\cdot, \cdot)$. In practice, typically the kernel will have some hyperparameters
$\theta$ that we must specify. For example, the radial basis function kernel takes the form

$$ k(\mathbf{x}_p, \mathbf{x}_q) = \sigma^2_f \exp\left(-\frac{1}{2\ell^2}(\mathbf{x}_p -  \mathbf{x}_q)^2\right) $$

where our hyperparameters $\theta = \{\sigma^2_f, \ell\}$ are the signal variance $\sigma^2_f$ and the length-scale $\ell$. Note that, with our assumption of i.i.d. Gaussian noise we then have 

$$ cov(\mathbf{x}_p, \mathbf{x}_q) = \sigma^2_f \exp\left(-\frac{1}{2\ell^2}(\mathbf{x}_p -  \mathbf{x}_q)^2\right) + \delta_{pq}\sigma^2 $$

These parameters can have a _major impact_ on our final predictions. Thus, we need a systematic way to chose "good" values of
our hyperparameters. One way to do so is to set the parameters via maximum likelihood estimation, i.e., we choose the parameters
that maximize the likelihood $p(\mathbf{y} \mid X)$ of our observed training data given the corresponding inputs. Based on our
assumptions in the previous subsection, we know

$$ \mathbf{y} \mid X \sim \mathcal{N}(\mathbf{0}, K(X, X) + \sigma_n^2I).$$

Thus, from the definition of the multivariate Gaussian distribution, we have

$$ p(\mathbf{y} \mid X) = -\frac{1}{2}\mathbf{y}^{T}(K(X, X) + \sigma_n^2I)^{-1}\mathbf{y} - \frac{1}{2}\left|K(X, X) + \sigma_n^2I\right| - \frac{n}{2}\log 2\pi $$