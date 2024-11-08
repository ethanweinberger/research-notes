# RNA Velocity

Due to technical limitations, single-cell RNA sequencing destroys individual cells. The RNA velocity model of La Manno et al. attempts to compute the time derivative of gene expression state by leveraging measurements of spliced and unspliced RNA counts.

---

RNA velocity assumes the following simplified model of transcription. First let $u$ denote the number of unspliced mRNAs for a given gene. We assume that the change in $u$ is governed by

```{math}
:label: unspliced
\frac{du}{dt} = \alpha - \beta u
```

Here $\alpha$ is interpreted as a (time-independent) transcription rate, while $\beta$ represents a (time-independent) splicing rate. Letting $s$ denote the corresponding number of mature/spliced mRNAs, we have 

```{math}
:label: spliced
\frac{ds}{dt} = \beta u - \gamma s
```

where $\gamma$ denotes the (time-independent) degradation rate of mature mRNAs. Because our ordinary differential equations described above are relatively simple (i.e., linear + first-order), we can solve them analytically. 

---

We begin with Equation {eq}`unspliced` for unspliced RNAs. To solve this equation, we'll proceed via [integrating factors](https://en.wikipedia.org/wiki/Integrating_factor). To do so, we need to rewrite our equation in the form:

```{math}
\frac{du}{dt} + P(t)u = Q(t)
```

By a simple rewriting of Equation {eq}`unspliced` we have


```{math}
:label: unspliced_rewritten
\frac{du}{dt} + \underbrace{\beta}_{P(t)} u = \underbrace{\alpha}_{Q(t)}
```

We then compute the integrating factor

```{math}
m(t) = e^{\int P(t) dt} = e^{\int \beta dt} = e^{\beta t}
```

Multiplying Equation {eq}`unspliced_rewritten` by our integrating factor we have

```{math}
e^{\beta t} \frac{du}{dt} + e^{\beta t} \beta u = e^{\beta t} \alpha
```

The LHS of this equation can be rewritten as

```{math}
e^{\beta t} \frac{du}{dt} + e^{\beta t} \beta u = \frac{d}{dt}\left(e^{\beta t}u\right)
```

Thus, 

```{math}
\frac{d}{dt}\left(e^{\beta t}u\right) = e^{\beta t} \alpha
```

Integrating both side with respect to $t$ and solving for $u$ yields

```{math}
:label: u_solved
u = \frac{\alpha}{\beta} + e^{-\beta t}C 
```

Now let's assume for our initial value condition we have $u(t) = u_0$. Plugging this in we have

```{math}
\begin{align}
&u_0 = \frac{\alpha}{\beta} + e^{0}C \\
\implies& C = u_0 - \frac{\alpha}{\beta}
\end{align}
```

Now plugging our expression for $C$ into Equation {eq}`u_solved` we obtain

```{math}
:label: u_final
u = u_0 e^{-\beta t} + \frac{\alpha}{\beta}(1 - e^{-\beta t})
```

---

We now proceed to solve for $s$. We first rewrite Equation {eq}`spliced` for $s$ as

```{math}
\begin{align}
\frac{ds}{dt} + \gamma s &= \beta u \\
&= \beta (u_0 e^{-\beta t} + \frac{\alpha}{\beta}(1 - e^{-\beta t})) \\
&= \beta u_0 e^{-\beta t} + \alpha (1 - e^{-\beta t})
\end{align}
```

where we plugged in our previous solution for $u$. Our integrating factor is then

```{math}
m(t) = e^{\int \gamma dt} = e^{\gamma t}
```
Multiplying by our integrating factor gives us

```{math}
e^{\gamma t}\frac{ds}{dt} + e^{\gamma t}\gamma s = e^{\gamma t}\left(\beta u_0 e^{-\beta t} + \alpha (1 - e^{-\beta t})\right)
```

Recognizing the LHS of this equation we have

```{math}
\frac{d}{dt}\left(e^{\gamma t}s\right) = e^{\gamma t}\left(\beta u_0 e^{-\beta t} + \alpha (1 - e^{-\beta t})\right)
```

Integrating both sides then yields 

```{math}
\begin{align}
e^{\gamma t}s &= \int e^{\gamma t}\left(\beta u_0 e^{-\beta t} + \alpha (1 - e^{-\beta t})\right)dt \\
&= \int \left(\beta u_0 e^{(\gamma - \beta)t} + \alpha e^{\gamma t} - \alpha e^{(\gamma -\beta)t} \right)dt \\
&= \frac{\beta u_0}{\gamma - \beta}e^{(\gamma - \beta)t} + \frac{\alpha}{\gamma} e^{\gamma t} - \frac{\alpha}{\gamma - \beta}e^{(\gamma - \beta)t} + C\\
&= \frac{\beta u_0 - \alpha}{\gamma - \beta}e^{(\gamma - \beta)t} + \frac{\alpha}{\gamma}e^{\gamma t} + C
\end{align}
```

Solving for $s$ then yields

```{math}
:label: s_solved
\begin{align}
s &= \frac{\beta u_0 - \alpha}{\gamma - \beta}e^{-\beta t} + \frac{\alpha}{\gamma} + e^{-\gamma t}C
\end{align}
```

Now, assuming the initial value condition $s(0) = s_0$ we have

```{math}
\begin{align}
s_0 &= \frac{\beta u_0 - \alpha}{\gamma - \beta}e^{-0} + \frac{\alpha}{\gamma} + e^{-0}C \\
\implies C &= s_0 - \left(\frac{\beta u_0 - \alpha}{\gamma - \beta} \right) - \frac{\alpha}{\gamma}
\end{align}
```

Plugging our expression for $C$ into Equation {eq}`s_solved` we then have 

```{math}
:label: s_final

s = e^{-\gamma t}s_0 + \frac{\alpha}{\gamma}(1 - e^{-\gamma t}) + \frac{\beta u_0 - \alpha}{\gamma - \beta}\left(e^{-\beta t} - e^{-\gamma t}\right)

```