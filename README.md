# Genetic Drift & Deep Learning: Wright–Fisher Baselines

**Phase 1: Drift Baselines** *Simulating evolutionary stochasticity using sequential neural networks.*

## 🧬 Project Overview
This project investigates whether deterministic deep learning models (**LSTM, GRU, Transformer**) can approximate the inherent randomness of **Genetic Drift** in finite populations. 

We simulate allele frequencies using the **Wright–Fisher model** and evaluate how well sequence models capture stochastic trajectories compared to the analytic ground truth provided by **Kimura’s Diffusion Theory**.

$$dp = s p(1-p) dt + \sqrt{\frac{p(1-p)}{2N_e}} dW_t$$

## 🧪 Methodology
1.  **Simulation**: Synthetic datasets generated via Wright–Fisher simulation (Binomial sampling). Data validity is verified against Kimura’s analytic variance ($V \approx p(1-p)/2N$).
2.  **Input**: Windowed history of allele frequencies ($p_{t-w:t}$) plus parameters $N$ and $s$.
3.  **Task**: Autoregressive forecasting of the next generation ($p_{t+1}$).
4.  **Models**:
    * **LSTM**: Standard recurrent baseline.
    * **GRU**: Gated Recurrent Unit (Streamlined RNN).
    * **Tiny Transformer**: Encoder-only attention model without sigmoid activations.

## 📊 Key Results

### 1. Short-Term Forecasting (RMSE)
Evaluated on single-step predictions ($t \to t+1$):
* **GRU (Best):** Achieved the lowest RMSE, indicating its gating mechanism best captures short-term stochastic structure.
* **LSTM:** Performed comparably to the GRU, demonstrating reliable learning.
* **Transformer:** Performed weakest; the architecture struggled with the low-dimensional, noise-dominated signal.
* *Note:* All models hit a "noise floor" ($\text{RMSE} \approx 0.03$) due to the irreducible randomness of the biological process.

### 2. Long-Term Drift Dynamics (Rollout)
Evaluated on recursive, multi-step forecasting:
* **Directionality:** All models correctly inferred the direction of drift driven by positive selection.
* **LSTM:** Generated the most realistic trajectories, tracking drift shape moderately without instability.
* **GRU:** Learned the strong upward trend but lacked probabilistic constraints, frequently overshooting biological boundaries ($p > 1$).
* **Transformer:** Underestimated selection strength, resulting in aggressively "oversmoothed" trajectories that failed to capture drift intensity.

## 🚀 Future Roadmap
* **Phase 2**: Implement **xLSTM** (Extended LSTM) for long-term stochastic stability.
* **Phase 3**: Introduce **TiRex** or probabilistic output heads to model Kimura variance explicitly rather than point estimates.

## 📦 Usage

**1. Requirements**
```bash
pip install numpy pandas matplotlib seaborn torch
