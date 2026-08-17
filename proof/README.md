# Theoretical Validation of the Adaptive Conformal Uncertainty Quantifier (ACUQ)

This directory contains the formal validity analysis of the ACUQ module proposed in
*"Uncertainty-Aware Online Ensemble Transformer for Accurate Cloud Workload Forecasting in Predictive Auto-Scaling"*.

## Which file to read

| File | Status |
| --- | --- |
| `Proof_Revised.pdf` | **Current version — read this one.** Revised statement of the coverage target and of the main theorem, with a complete self-contained proof. |
| `Proof_Revised.tex` | LaTeX source of `Proof_Revised.pdf` (compiles with `pdflatex`, no external `.bib`). |
| `Proof_v1_superseded.pdf` | First version, kept only for provenance. Its theorem statement contains the error described in the next section. |

## What was corrected, and why

All three items below were errors in how the results were *written*; the algorithm
(Eq. (18) and Algorithm 1 of the paper) and the implementation in `../code/` are unchanged,
so no reported experimental number is affected.

1. **Coverage target (Eq. (1) of the paper).** The indicator function was placed inside the
   probability and the symbol $t$ was overloaded as both the current timestamp and the
   lead-time index. The intended full-horizon (simultaneous) event is

   $$\mathbb{P}\bigl(x_{t+j}\in c_{t+j},\ \forall j\in\{1,\dots,H\}\bigr)\ \ge\ 1-\alpha,
   \qquad\text{i.e.}\qquad \mathbb{P}(Y_t\in C_t)\ \ge\ 1-\alpha,$$

   with $C_t=c_{t+1}\times\cdots\times c_{t+H}$ a box of $H$ intervals.

2. **Empirical copula constraint (Eq. (15)–(16) of the paper).** Coverage is a *lower-orthant*
   event $u_t\preceq u^{*}$, so the quantity to control is

   $$\widehat{Q}(u)=\frac{1}{|\mathcal{L}_{\mathrm{cop}}|+1}\sum_{i\in\mathcal{L}_{\mathrm{cop}}}\prod_{j=1}^{H}\mathbf{1}\{u^{i}_{j}\le u_{j}\},
   \qquad u^{*}=\arg\min_{u}\sum_{j=1}^{H}u_j\ \ \text{s.t.}\ \ \widehat{Q}(u^{*})\ge 1-\alpha_t .$$

   The paper displayed the reversed comparison together with the constraint
   $\widehat{Q}(u^{*})\le\alpha_t$; these two conditions are mutually inconsistent, and the
   lower-orthant form above is the one used in the proof, in Copula Conformal Prediction,
   and in the implementation.

3. **Main theorem.** The previous statement used $\prod_{j=1}^{H}\mathbf{1}\{x^{j}_{t}\notin C^{j}_{t}\}$,
   i.e. the event that *all* $H$ future values fall outside their intervals. For $H>1$ this is
   **not** the complement of the full-horizon coverage event in item 1: the complement is that
   *at least one* lead time is violated. The two coincide only when $H=1$.

## Corrected main result

Let $\mathrm{err}_t:=\mathbf{1}\{Y_t\notin C_t\}=1-\prod_{j=1}^{H}\mathbf{1}\{x_{t+j}\in c_{t+j}\}$
be the joint miscoverage indicator, which is exactly what Eq. (18) and Algorithm 1 already
monitor. **Theorem 4** (Sec. 2 of `Proof_Revised.pdf`; the corrected form of Theorem 2 of the paper) states

$$\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T}\Bigl(1-\prod_{j=1}^{H}\mathbf{1}\{x_{t+j}\in c_{t+j}\}\Bigr)=\alpha ,$$

equivalently, the full-horizon event of Eq. (1) occurs with long-run frequency $1-\alpha$.
The convergence is pathwise with an explicit finite-$T$ rate,

$$\Bigl|\tfrac{1}{T}\textstyle\sum_{t=1}^{T}\mathrm{err}_t-\alpha\Bigr|\ \le\ \frac{\max\{\alpha_1,1-\alpha_1\}+\gamma_Q}{\gamma_Q\,T}\ =\ O\!\left(\frac{1}{\gamma_Q T}\right),$$

and requires no assumption on the data-generating process — in particular neither
exchangeability nor stationarity.

## Two complementary guarantees

The revised note separates them explicitly, which the earlier version did not:

- **Finite-sample joint validity** (Proposition 8): $\mathbb{P}(Y_t\in C_t)\ge1-\alpha$ at a fixed
  level, under exchangeability of the nonconformity scores. This is the copula-conformal ingredient.
- **Long-run joint validity** (Theorem 4): the time-average result above, which drops
  exchangeability by replacing $\alpha$ with the adaptive level $\alpha_t$.

ACUQ therefore provides no per-step marginal or conditional guarantee; only the two statements
above are claimed.

## Structure of `Proof_Revised.pdf`

| Section | Contents |
| --- | --- |
| 1 | Setting, notation, the joint miscoverage indicator, the corrected empirical copula, and the saturation conventions (C1)–(C2) |
| 2 | The corrected theorem statement, plus a counterexample showing why the previous statement did not imply Eq. (1) |
| 3 | Finite-sample joint validity at a fixed level (needs exchangeability) |
| 4 | Boundedness of $\alpha_t$ and the proof of the main theorem (no exchangeability) |
| 5 | Scope of the guarantee, and a per-horizon (marginal) variant |
| 6 | The resulting list of edits to the paper |

The saturation conventions matter for correctness: because
$\widehat{Q}(\mathbf{1})=|\mathcal{L}_{\mathrm{cop}}|/(|\mathcal{L}_{\mathrm{cop}}|+1)<1$, the
constraint $\widehat{Q}(u^{*})\ge1-\alpha_t$ becomes infeasible once
$\alpha_t<1/(|\mathcal{L}_{\mathrm{cop}}|+1)$, and this band is genuinely reachable under the
online update. With the settings of the paper ($|\mathcal{L}_t|=60$, $\gamma_Q=0.05$, $\alpha=0.1$),
two consecutive miscoverages from $\alpha_1=0.1$ give $\alpha_3=0.01<1/31$. In that case, as
when $\alpha_t\le0$, the construction returns $u^{*}=\mathbf{1}$ and $C_t=\mathbb{R}^{H}$,
which is the conservative direction required by the argument.

## References

The derivation builds on conformal prediction [Vovk et al. 2005, 2017], copula conformal
prediction for multi-step forecasting [Sun and Yu, ICLR 2024], and adaptive conformal inference
under distribution shift [Gibbs and Candès, NeurIPS 2021]. Full citations are in the note.
