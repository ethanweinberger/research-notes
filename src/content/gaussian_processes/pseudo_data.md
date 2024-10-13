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