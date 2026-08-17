# Uncertainty-Aware Online Ensemble Transformer: Supplementary Materials
This repository contains supplementary materials for the paper *"Uncertainty-Aware Online Ensemble Transformer for Accurate Cloud Workload Forecasting in Predictive Auto-Scaling"*, including theoretical proofs and source code implementations.


## Document Structure
The repository is organized into two core directories to separate theoretical and practical components:

```
├── proof/                        # Theoretical proofs and mathematical derivations
│   ├── README.md                 # What was corrected, and which file is authoritative
│   ├── Proof_Revised.pdf         # Current formal proof of the uncertainty quantifier's validity
│   ├── Proof_Revised.tex         # LaTeX source of the above
│   └── Proof_v1_superseded.pdf   # Superseded first version, kept for provenance
└── code/                  # Source code implementation
    ├── ...
    ├── main.py            # Core experiment configuration and execution
    ├── requirements.txt   # Dependencies for environment setup
    ├── run.sh             # Script for basic forecasting experiments
    └── run_transfer.sh    # Script for transfer learning experiments

```


## Proof: Theoretical Validation
The `proof/` directory contains the theoretical foundation of the proposed framework. The current
version is **`proof/Proof_Revised.pdf`**, which validates the **Adaptive Conformal Uncertainty
Quantifier (ACUQ)**; see [`proof/README.md`](proof/README.md) for what was corrected relative to
the first version and why. The corrections concern the *statements* only — the algorithm
(Eq. (18) and Algorithm 1) and the implementation in `code/` are unchanged, so no reported
experimental number is affected.

### Key Theoretical Results
1. **Long-run full-horizon validity (Theorem 4, the corrected form of Theorem 2 of the paper).**
   With the joint miscoverage indicator
   $\mathrm{err}_t=\mathbf{1}\{Y_t\notin C_t\}=1-\prod_{j=1}^{H}\mathbf{1}\{x_{t+j}\in c_{t+j}\}$,
   i.e. the event that *at least one* lead time is violated,

   $$\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T}\Bigl(1-\prod_{j=1}^{H}\mathbf{1}\{x_{t+j}\in c_{t+j}\}\Bigr)=\alpha ,$$

   so the full-horizon coverage event of Eq. (1) occurs with long-run frequency $1-\alpha$. The
   convergence is pathwise with an explicit finite-$T$ rate,
   $\bigl|\frac{1}{T}\sum_{t\le T}\mathrm{err}_t-\alpha\bigr|\le(\max\{\alpha_1,1-\alpha_1\}+\gamma_Q)/(\gamma_Q T)$,
   and needs no assumption on the data-generating process — neither exchangeability nor stationarity.

2. **Finite-sample joint validity (Proposition 8).** At a fixed level,
   $\mathbb{P}(Y_t\in C_t)\ge1-\alpha$ under exchangeability of the nonconformity scores. This is
   the copula-conformal ingredient, complementary to result 1.

3. **Validity of the marginal predictive distributions (Lemma 7).** The randomised CDF built from
   the calibration split satisfies $\mathbb{P}_{Z}[\hat{F}_{j}(s_{j})\le 1-\alpha]=1-\alpha$ for any
   $0<\alpha<1$.

4. **Boundedness of the adaptive level (Lemma 9).** $\alpha_t\in[-\gamma_Q,\,1+\gamma_Q]$ for all
   $t$, which is what makes the telescoping bound in result 1 valid.

ACUQ provides no per-step marginal or conditional guarantee; only the statements above are claimed.

### Proof Methodology
The derivation builds on conformal prediction [Vovk et al. (2005, 2017)], copula conformal
prediction for multi-step forecasting [Sun and Yu (2024)], and adaptive conformal inference under
distribution shift [Gibbs and Candès (2021)]. Joint (full-horizon) coverage is reduced to a
lower-orthant dominance event in the unit cube via an empirical copula; exchangeability is then
required only for the finite-sample result, while the long-run result follows from a deterministic
telescoping argument on the online update of the adaptive miscoverage rate.


## Code: Implementation Details
The `code/` directory provides a complete implementation of the framework, supporting:
- Multiple state-of-the-art forecasting models (e.g., `onenet_fsnet`, `timesnet`, `dlinear`)
- Flexible configuration of sequence lengths, prediction horizons, and training parameters
- Online learning mode and transfer learning experiments
- Automated result saving for metrics, predictions, and ground truths

