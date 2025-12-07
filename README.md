# Genetic Drift & Deep Learning: Wright–Fisher Baselines

**Phase 1: Drift Baselines** *Simulating evolutionary stochasticity using sequential neural networks.*

## 🧬 Project Overview
This project investigates whether deterministic deep learning models (**LSTM, GRU, Transformer**) can approximate the inherent randomness of **Genetic Drift** in finite populations. 

We simulate allele frequencies using the **Wright–Fisher model** and evaluate how well sequence models capture stochastic trajectories compared to the analytic ground truth provided by **Kimura’s Diffusion Theory**.

$$dp = s p(1-p) dt + \sqrt{\frac{p(1-p)}{2N_e}} dW_t$$

## 🧪 Methodology
1.  **Simulation**: Synthetic datasets generated via Wright–Fisher simulation (Binomial sampling).
2.  **Input**: Windowed history of allele frequencies ($p_{t-w:t}$) plus parameters $N$ and $s$.
3.  **Task**: Autoregressive forecasting of the next generation ($p_{t+1}$).
4.  **Models**:
    * **LSTM**: Standard recurrent baseline.
    * **GRU**: Gated Recurrent Unit (Streamlined RNN).
    * **Tiny Transformer**: Encoder-only attention model without sigmoid activations.

## 📊 Key Results
* **Analytic Benchmark**: The simulation accurately matches Kimura’s theoretical variance ($V \approx p(1-p)/2N$).
* **Best Model**: The **GRU** captured short-term drift structure most effectively (lowest RMSE).
* **Transformer Issues**: The Transformer struggled with the low-dimensional, high-noise data, resulting in over-smoothed predictions.
* **The Stochastic Limit**: All deterministic models hit a "noise floor" (RMSE $\approx 0.03$) due to the irreducible randomness of the biological process.

## 🚀 Future Roadmap
* **Phase 2**: Implement **xLSTM** (Extended LSTM) for long-term stochastic stability.
* **Phase 3**: Introduce **TiRex** or probabilistic output heads to model distributions rather than point estimates.

## 📦 Usage

**1. Requirements**
```bash
pip install numpy pandas matplotlib seaborn torch
