# Gaussian Processes 3: Sparse GPs with inducing points

In the previous section we discussed how to perform Gaussian process regression to estimate the
values of a function $f$. Our solution relied on the convenient properties of the multivariate
normal distribution, which allowed us to compute posterior distributions of unknown function outputs
in closed form.

Unfortunately, while it's straightforward to write down the equations for the posterior distribution,
actually _computing_ it is non-trivial. Specifically, the posterior distribution requires inverting
the covariance matrix $K(X, X)$, which requires $\mathcal{O}(N^3)$ operations; thus as the size of our training
data increases, this operation quickly becomes infeasible.

To work around this issue, a number of works have proposed so-called _sparse_ Gaussian process methods that take varying
approaches to approximate the information contained in the full covariance matrix. In this section we'll review one popular
sparse GP method, known as the _inducing point_ method.

--- 

When placing a GP prior on a function, we implicitly assert the existence of some redundancy and/or correlation in the relationship between the function's inputs and outputs. For example, the RBF kernel

$$k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\sigma^2}\right)$$

enforces that function outputs corresponding to similar inputs should observe a high degree of correlation. For large training datasets, we can then imagine that it might be possible to remove some points from the dataset, thus reducing our computational burden, without having a great impact on our predictions.

Taking this idea one step further, we can imagine that it might be possible to design some ideal "pseudo-dataset" $\bar{\mathcal{D}} = \{\mathbf{x}_m, \bar{f}_m\}_{m=1}^{M}$ with $M << N$ that captures a similar amount of information as the full dataset. Note that here for our pseudo-dataset we have intentially used $\bar{f}$ to describe our output values rather than $\bar{y}$, as it doesn't make much sense to add noise to output function values that we're choosing ourselves.

---

Assuming that our pseudo-dataset can take the place of the full dataset, an output $y$ has the likelihood

$$ p(y \mid \mathbf{x}, \bar{X}, \bar{\mathbf{f}}) = \mathcal{N}(k_{\mathbf{x}}^{T}K_{\bar{X}\bar{X}}^{-1}\bar{\mathbf{f}}, K_{\mathbf{x}\mathbf{x}} - k_{\mathbf{x}}^{T}K_{\bar{X}\bar{X}}^{-1}k_{\mathbf{x}} + \sigma^2),$$

where $k_{\mathbf{x}} \in \mathbb{R}^{M}$ is shorthand for a vector with entries $k(\mathbf{x}, \mathbf{x}_m)$. Assuming that our full set of test data $\mathbf{y} = \{y\}_{n=1}^{N}$, are i.i.d., $\mathbf{y}$ then has the following likelihood:

```{math}
:label: likelihood
p(\mathbf{y} \mid X, \bar{X}, \bar{\mathbf{f}}) = \prod_{n=1}^{N} p(y_n \mid \mathbf{x}_n, \bar{X}, \bar{\mathbf{f}}) = \mathcal{N}(\mathbf{y} \mid K_{NM}K_{M}^{-1}\bar{\mathbf{f}}, \mathbf{\Lambda} + \sigma^2I),
```

where $\mathbf{\Lambda} = \text{diag}(\mathbf{\lambda})$, and $\lambda_n = K_{\mathbf{x}_n\mathbf{x}_n} - k_{\mathbf{x}_n}^{T}K_{M}^{-1}k_{\mathbf{x}_n}$. With our model specified, we must now resolve two issues. First, we neeed to specify some criterion for choosing a "good" pseudo-dataset.

One way to do so would be to optimize $\bar{\mathbf{f}}$ and $\bar{X}$ to maximize the above likelihood. However, by treating $\bar{\mathbf{f}}$ with some uncertainty we can actually make our lives easier and avoid the need to optimize this quantity altogether. To ensure that the pseudo data outputs model the true dataset well, it's reasonable to assume that that they follow the same prior as true data points, i.e.,

```{math}
:label: pseudo_prior
p(\bar{\mathbf{f}} \mid \bar{X}) = \mathcal{N}(0, K_{\bar{X}\bar{X}}).
```

With this assumption, we can then marginalize out $\bar{\mathbf{f}}$ from Equation {eq}`likelihood`. I.e., we compute

$$ p(\mathbf{y} \mid X, \bar{X}) = \int p(\mathbf{y} \mid X, \bar{X}, \bar{\mathbf{f}})p(\bar{\mathbf{f}} \mid \bar{X})d\bar{\mathbf{f}}, $$

From the properties of Gaussians this integral has a closed form and results in

$$ p(\mathbf{y} \mid X, \bar{X}) = \mathcal{N}(\mathbf{y} \mid 0, K_{NM}K_{M}^{-1}K_{MN} + \mathbf{\Lambda} + \sigma^2I).$$

And so we can find a good set of pseudo-inputs $\bar{X}$ by maximizing the above expression for the marginal likelihood. Next, we need a way to make predictions on the distribution of unseen points given our observed data. In other words, we need a method for computing $p(y_* \mid \mathbf{x}_*, X, \bar{X}, \mathbf{y})$. Of particular note, we can't simply reuse our result from Equation {eq}`likelihood`, as that expression depends on the pseudo-outputs $\bar{\mathbf{f}}$, which we chose not to learn. Thus, we must instead compute

```{math}
:label: predictive_dist
p(y_* \mid \mathbf{x}_*, X, \bar{X}, \mathbf{y})  = \int p(y_* \mid \mathbf{x}_*, X, \bar{X}, \mathbf{y}, \bar{\mathbf{f}})p(\bar{\mathbf{f}} \mid X, \bar{X}, \mathbf{y})d\mathbf{\bar{f}}.
```

The first term in this integral is essentially Equation {eq}`likelihood`, and thus is straightforward to compute. To obtain the second term we can apply Bayes rule with Equations {eq}`likelihood` and {eq}`pseudo_prior` to obtain:

$$ p(\bar{\mathbf{f}} \mid X, \bar{X}, \mathbf{y}) = \mathcal{N}(\bar{\mathbf{f}} \mid K_{\bar{X}\bar{X}}Q^{-1}K_{\bar{X}X}(\mathbf{\Lambda} + \sigma^2I)\mathbf{y}, K_{\bar{X}\bar{X}}Q^{-1}K_{\bar{X}\bar{X}}),$$

where $Q = K_{\bar{X}\bar{X}} + K_{\bar{X}X}(\mathbf{\Lambda} + \sigma^2I)K_{X\bar{X}}$.

Plugging this expression into Equation {eq}`predictive_dist` we have (after some tedious computations leveraging the properties of Gaussians):

$$
\begin{align}
p(y_* \mid \mathbf{x}_*, X, \bar{X}, \mathbf{y}) &= \mathcal{N}(y_* \mid \mu_*, \sigma^2_*), \\
\mu_* &= k_{\mathbf{x}_*}^{T}Q^{-1}K_{\bar{X}X}(\mathbf{\Lambda} + \sigma^2I)^{-1}\mathbf{y}, \\
\sigma_*^2 &= K_{\mathbf{x}_*\mathbf{x}_*} - k_{\mathbf{x}_*}^{T}(K_{\bar{X}\bar{X}}^{-1} - Q^{-1})k_{\mathbf{x}_*} + \sigma^2
\end{align}
$$

Before moving on, let's briefly consider the cost of computing the above distribution. Calculating $Q$ requires multiplying a matrix with dimensions $M \times N$ and a matrix with dimensions $N \times M$, and thus this step is of order $\mathcal{O}(M^2N)$. As they are both $M \times M$ matrices, computing the inverses of $Q$ and $K_{\bar{X}\bar{X}}$ requires $\mathcal{O}(M^3)$ steps. Notably, the remaining inverse computation for $\mathbf{\Lambda} + \sigma^2I$ is trivial, as this matrix is diagonal. As we take $M << N$, our prediction step is thus dominated by the $\mathcal{O}(M^2N)$ term, which is indeed a major improvement on the  $\mathcal{O}(N^3)$ computations required for exact GP regression.
