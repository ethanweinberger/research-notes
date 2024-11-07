# The exponential family

We begin with a definition.

````{prf:definition}
:label: exponential_family

For a given parameter $\eta$, an _exponential family_ of distributions is a set of probability distributions whose densities can be written as 

$$ p(x \mid \eta) = h(x)\exp\{\eta^{T}T(x) - A(\eta)\} $$
````

It turns out that many important probability distributions can be written in this way. We now proceed to give some examples. 

````{prf:example}
:label: exponential_family_bernouli

A Bernoulli random variable $X$ with parameter $\pi$ assigns a probability measure $\pi$ to $x=1$ and $1 - \pi$ to $x = 0$. This corresponds to the density

```{math}
\begin{align}
p(x \mid \pi) &= \pi^{x}(1 - \pi)^{1 - x} \\
&= \exp \left\{\log\left(\frac{\pi}{1-\pi}\right)x + (1 - \pi)\right\}
\end{align}
```

Thus, Bernoulli random variables comprise an exponential family of distributions with

```{math}
\begin{align}
h(x) &= 1 \\
\eta &= \log\left(\frac{\pi}{1 - \pi}\right) \\
T(x) &= x \\
A(\eta) &= \log (1 + e^{\eta}),
\end{align}
```

where $A(\eta)$ can be found by solving $\log (1 - \pi) = -A(\eta)$.
````

Now let's do something a little more complicated.

````{prf:example}
:label: exponential_family_poisson

A Poisson random variable $X$ with parameter $\lambda$ follows the density

```{math}
\begin{align}
p(x \mid \lambda) &= \frac{\lambda^x e^{-\lambda}}{x!} \\
&= \exp\left\{\log\left(\frac{\lambda^x e^{-\lambda}}{x!}\right)\right\} \\
&= \frac{1}{x!}\exp\left\{\log(\lambda)x - \lambda\right\}
\end{align}
```

Thus, Poisson random variables form an exponential family of distributions with

```{math}
\begin{align}
h(x) &= \frac{1}{x!} \\
\eta &= \log(\lambda) \\
T(x) &= x \\
A(\eta) &= e^{\lambda}
\end{align}
```

````