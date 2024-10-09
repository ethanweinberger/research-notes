# Gaussian Processes 1: Preliminaries

We begin with a general definition.

````{prf:definition}
:label: stochastic-process

A stochastic process $f$ is a collection of random variables indexed by $x \in \mathcal{X}$. I.e., 

$$ f = \{ f(x) : x \in \mathcal{X}\} $$

````

In the above definition, $\mathcal{X}$  is an arbitrary indexing set. When modeling real-world
phenomena, we typically assume that $\mathcal{X}$ corresponds to some semantically meaningful concept. For example,
if we want to model some phenomenon over _time_, we could use $\mathcal{X} = \mathbb{R}$ with $x \in \mathcal{X}$
corresponding to an individual point in time. Similarly, if we want to model some phenomenon that varies across
two-dimensional _space_ we could let $\mathcal{X} = \mathbb{R}^2$ with $x \in \mathcal{X}$ now corresponding
to spatial coordinates.

If $\mathcal{X} = \mathbb{R}^n$, then we say that $f$ is an infinite-dimensional process. Based on the examples above,
it's clear that we'd like to be able to model infinite-dimensional processes! However, dealing with infite collections
of random variables presents some technical mathematical difficulties. For example, can we define the law of
law of $f$? I.e., can we compute statements like $\mathbb{P}(f(x_1) \in [a_1, b_1], f(x_2) \in [a_2, b_2], \ldots)$?. Is
this law guaranteed to be unique? Etc.


Fortunately for us, Kolmogorov [showed](https://en.wikipedia.org/wiki/Kolmogorov_extension_theorem) that we can get
away with only considering finite-dimensional distributions.

````{prf:definition}
:label: fdds

For a stochastic process $f$ we define $f$'s finite-dimensional distributions (FDDs) as the collection of distributions

$$
    \mathbb{P}(f(x_1) \leq y_1, \ldots, f(x_n) \leq y_n)
$$

for all finite sets $(x_1, \ldots, x_n)$ of indices in $\mathcal{X}$,

````

In particular, for a given process $f$, the FDDs uniquely determine the law of $f$. This brings us to our central object of study,
the _Gaussian process_.

````{prf:definition}
:label: gaussian-process

A Gaussian process (GP) is a stochastic process with Gaussian finite dimensional distributions. I.e., 

$$ (f(x_1), \ldots, f(x_n)) \sim \mathcal{N}(\mu, \Sigma)$$

A GP is completely specified by its mean and covariance, which specify as functions of the index set. For a GP $f$ with
mean function $m(x)$ and covariance function $k(x, x')$, we write

$$f \sim \mathcal{GP}(m(x), k(x, x'))$$
````

For computational convenience we'll typyically take $m(x) = 0$. For the covariance, we can choose any positive
semidefinite function, and we'll typically choose $k$ to reflect some prior knowledge. For example, if we expect output values
to vary smoothly across time, we'll choose $k$ to reflect this fact. We defer a detailed discussion on kernel functions until later. 

Now, _why_ are we considering the Gaussian process specifically? In short, the answer lies
in the many convenient properties of multivariate Gaussians. For example, sums of Gaussians are Gaussian, and the marginal
distributions of a multivariate Gaussian are Gaussian. In particular, one useful property of Gaussians is that they're closed
under conditioning.

```{prf:proposition}
:label: gp-conditioning

Let $\mathbf{f}$ denote the output of $f \sim \mathcal{GP}(\mathbf{0}, k(x, x'))$ at a set of training inputs $X$, and define $\mathbf{f_*}$ correspondingly for a set of test inputs whose values we don't observe. We then have the joint distribution

$$ \begin{pmatrix} \mathbf{f} \\ \mathbf{f_*} \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0},\begin{bmatrix} K(X,X), K(X, X_*) \\ K(X_*, X), K(X_*, X_*)\end{bmatrix}\right)$$

Conditioning on the observed training points then gives us

$$\mathbf{f_*} \mid X_*, X, \mathbf{f} \sim \mathcal{N}(K(X_*, X)K(X, X)^{-1}\mathbf{f}, K(X_*, X_*) - K(X_*, X)K(X, X)^{-1}K(X, X_*))$$
```

The above proposition is _extremely_ useful. By specifying some prior on how our function's outputs should be have with respect to the inputs (i.e., the covariance function $k$), we can leverage any observed data points to make predictions on the distributions for points at unobserved inputs. 