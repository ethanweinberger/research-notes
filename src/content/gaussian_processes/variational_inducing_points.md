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

\newcommand\z{\mathbf{z}}
\newcommand\y{\mathbf{y}}
\newcommand\f{\mathbf{f}}
\newcommand\x{\mathbf{x}}
```


# Gaussian Processes 4: Variational Learning of Inducing Variables in Sparse Gaussian Processes

Last time we introduced the idea of pseudo-data/inducing points for scalable GP learning. In the original formulation of Snelson and Gharamani (2009), this was done by maximizing the log marginal likelihood

$$\log p(\y) = \log \mathcal{N}(\y \mid \boldsymbol{0}, Q_{nn} + \sigma^2I),$$

where $Q_{nn}$ is an approximation to the true covariance matrix $K_{nn}$. While this method demonstrated some initial impressive results, it comes with some noticeable downsides. Notably, by optimizing with respect to a modified GP prior, we may not end up approximating the true exact GP model. To avoid this issue, we can instead follow a different approach. Rather than modifying the exact GP model, we'll instead try to minimize the _distance_ between the exact GP posterior and a variational approximation.

---

Our goal is to define a sparse method that approximates the exact (but computationally expensive) posterior GP mean and covariance functions. For a set of function values $\z$ and observed (noisy) function values $\y$ corresponding to true function values $\f$, we can write the exact posterior as:

$$ p(\z \mid \y) = \int p(\z \mid \f)p(\f \mid \y)d\f $$

Now suppose we wish to approximate the above computation via a set of inducing variables $\f_m$ representing function evaluations at inducing inputs $X_m$. To ensure that our inducing points correspond to our true function, we assume that they're drawn from the same prior as the training data points. With these additional points, our full model has the distribution

$$p(\y, \z, \f_m, \f) = p(\y \mid \z, \f_m, \f)p(\z, \f_m, \f)  = p(\y \mid \f)p(\z, \f_m, \f),$$

where the first equality comes from the chain rule of probability and the second equality uses the fact that $\y$ is simply a noisy version of $\f$ (and thus is independent of $\f_m$ and $\z$). We can then write

$$ p(\z \mid \y) = \int p(\z \mid \f_m, \f)p(\f \mid \f_m, \y)p(\f_m \mid \y)d\f d\f_m.$$

In an ideal setting, we want our inducing points to fully capture our functions behavior. To formalize this idea, we make the assumption that $p(\z \mid \f_m, \f) = p(\z \mid \f_m)$. That is, for a function value $\z$, $\f$ does not provide any information beyond that already provided by our inducing points. We can then rewrite our integral as:

$$ p(\z \mid \y) = \int p(\z \mid \f_m)p(\f \mid \f_m, \y)p(\f_m \mid \y)d\f d\f_m.$$

Moreover, from our assumption we also have $p(\f \mid \f_m, \y) = p(\f \mid \f_m)$, as $\y$ is simply a noisy version of $\f$ (and adding independent noise won't change any independence relationships), and our inducing point assumption applies to _all_ function values. We then have

```{math}
\begin{align}
p(\z \mid \y) &= \int p(\z \mid \f_m)p(\f \mid \f_m)p(\f_m \mid \y)d\f d\f_m \\
&= \int p(\z \mid \f_m)p(\f_m \mid \y)\left(\int p(\f \mid \f_m)d\f\right)d\f_m \\
&= \int p(\z \mid \f_m)p(\f_m \mid \y)d\f_m,
\end{align}
```

where the integral of the density in the second line evaluates to 1 for any fixed value of $\f_m$ and thus can be ignored. Now, the above expression applies in the scenario when we indeed know that $\f_m$ is a sufficient statistic for $\z$. In practice, it's highly nontrivial to find a set of points $\f_m$ that satisfies this property.

---

To work around this issue, we'll adopt a variational approach that will allow us to specify a rigorous procedure for selecting the parameters $(\boldsymbol{\mu}, A)$ for $\phi$ along with the locations of our inducing points $X_m$.

Recall that our goal is to learn a distribution $q(\cdot)$ that follows our inducing point assumption which is as close as possible to the true posterior $p(\cdot \mid \y)$. Given the training data available to us, this implies that we should minimize the distance between $q(\f)$ and $p(\f \mid \y)$. In other words we want to find

$$q^{*}(\f) = \arg\min_{q\in\mathcal{Q}}(q(\f) \mid\mid p(\f \mid \y))$$

Equivalently, if we augment our training data with a set of inducing outputs $\f_m$ evaluated at the inducing points $X_m$, we can solve

$$ \arg\min_{q\in\mathcal{Q}}(q(\f, \f_m) \mid\mid p(\f, \f_m \mid \y)).$$

Following standard variational inference procedure, we do so by maximizing the variational lower bound

```{math}
\begin{align}
F_{V}(X_m, \phi) &= \int q(\f, \f_m) \log \frac{p(\y, \f, \f_m)}{q(\f, \f_m)}d\f d\f_m \\
&= \int p(\f \mid \f_m) \phi(\f_m) \log \frac{p(\y \mid \f)p(\f\mid \f_m)p(\f_m)}{p(\f \mid \f_m)\phi(\f_m)}d\f d\f_m \\
&= \int p(\f \mid \f_m) \phi(\f_m) \log \frac{p(\y \mid \f)p(\f_m)}{\phi(\f_m)}d\f d\f_m \\
&= \int \phi(\f_m) \left(\int p(\f \mid \f_m)\log p(\y \mid \f)d\f + \log \frac{p(\f_m)}{\phi(\f_m)}\right)d\f_m
\end{align}
```

We now proceed as follows. First, we define

$$ \log G(\f_m, \y) = \int p(\f \mid \f_m)\log p(\y \mid \f)d\f$$

We then compute

```{math}
\DeclareMathOperator{\Tr}{Tr}
\begin{align}
\log G(\f_m, \y) &= \int p(\f \mid \f_m)\log p(\y \mid \f)d\f \\
&= \int p(\f \mid \f_m)\left(-\frac{n}{2}\log (2\pi\sigma^2) - \frac{1}{2\sigma^{2}} \Tr\left[\y\y^{T} - 2\y\f^{T} + \f\f^{T} \right]\right)d\f \\
&= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\Tr\left[\y\y^{T} - 2\y\boldsymbol{\mu}_{\f \mid \f_m}^{T} + \boldsymbol{\mu_{\f \mid \f_m}}\boldsymbol{\mu_{\f \mid \f_m}}^{T} + \boldsymbol{\Sigma}_{\f \mid \f_m}\right] \\
&= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\Tr\left[\y\y^{T} - 2\y\boldsymbol{\mu}_{\f \mid \f_m}^{T} + \boldsymbol{\mu_{\f \mid \f_m}}\boldsymbol{\mu_{\f \mid \f_m}}^{T}\right] + \Tr\left[\boldsymbol{\Sigma}_{\f \mid \f_m}\right] \\
&= \log \mathcal{N}(\y \mid \mu_{\f \mid \f_m}, \sigma^2I) - \frac{1}{2\sigma^2}\Tr\left[\boldsymbol{\Sigma}_{\f \mid \f_m}\right]
\end{align}
```

We then have

$$ F_{V}(X_m, \phi) = \int \phi(\f_m)\log\frac{G(\f_m, \y)p(\f_m)}{\phi(\f_m)}d\f_m - \frac{1}{2\sigma^2}\Tr(K_{nn} - Q_{nn})$$


