# TreeMemoryPredictor 🧠

A lightweight, adaptive, and explainable sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** is designed to predict the next token in a **discrete sequence** by learning patterns "on the fly." Unlike Neural Networks, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to changing data distributions.

It is particularly effective for:
*   **Stream Processing:** IoT sensors, user behavior analytics.
*   **Game AI:** Real-time opponent modeling.
*   **Data Compression:** PPM-style logic for custom formats.
*   **Bioinformatics:** DNA sequence analysis and motif discovery.
*   **Code Assistants:** Local, privacy-first autocompletion.

---

## ✨ Key Features

1.  **Dynamic Suffix Trie:** Builds a memory structure of variable-length contexts (from $N_{min}$ to $N_{max}$) in real-time.
2.  **Entropy Scaling ($S^L$):** Automatically adjusts pattern weights based on vocabulary size.
    *   *Result:* Long, precise matches in complex alphabets overpower noise automatically.
3.  **Lazy Exponential Decay:** Implements a "forgetting" mechanism ($Count \times Decay^{\Delta t}$). Old, unused patterns fade away, allowing the model to adapt to non-stationary data.
4.  **Batch & Stream Learning:** Supports both continuous streams and batched independent sequences (resetting context between items).
5.  **Log-Space Math:** All internal calculations are done in logarithmic space to prevent numerical overflow even with deep contexts ($N=100+$).

---

## 🚀 Quick Start

### Installation
Simply copy `tmp.py` (or rename it to `memory_predictor.py`) into your project. There are no heavy dependencies.

### 1. Basic Usage (Continuous Stream)

```python
from tmp import TreeMemoryPredictor

# Initialize: Context up to 6 steps, Decay 0.99
model = TreeMemoryPredictor(n_max=6, decay=0.99)

sequence = [0, 1, 0, 1, 0, 1, 0]

# Online Learning
for token in sequence:
    prediction = model.predict() # Returns 0 or 1
    model.update(token)          # Learn & Move window
```

### 2. Batch Training (Independent Sequences)

If you have a dataset of independent examples (e.g., user sessions, DNA strands, separate sentences), pass them as a **list of lists**. The model will reset its short-term memory (context) between sequences but keep long-term memory (weights).

```python
# Dataset: List of lists
dataset = [
    ['user', 'login', 'click', 'logout'],
    ['user', 'login', 'purchase', 'logout'],
    ['admin', 'login', 'delete', 'logout']
]

model = TreeMemoryPredictor(n_max=5)

# Fit on batch (Context resets automatically between rows)
model.fit(dataset) 

# Predict for a new session
model.fill_context(['user', 'login'])
print(model.predict()) # -> 'click' or 'purchase'
```

---

## 🔬 How It Works

The algorithm uses a **weighted voting system** across multiple Markov orders.

### 1. Context Mixing
For a sequence `... A B C`, the model looks up:
*   `C` (Order 1)
*   `B C` (Order 2)
*   `A B C` (Order 3)

### 2. The Weighting Formula (Log-Space)
Each node in the Trie calculates its "voting power" using this formula:

$$
\log(Weight) = \log(N_{obs}) + \Delta t \cdot \log(Decay) + L \cdot \log(S)
$$

Where:
*   **$N_{obs}$**: Number of times this pattern was observed.
*   **$Decay$**: Forgetting factor (e.g., 0.99).
*   **$\Delta t$**: Time steps since this node was last visited.
*   **$S$**: Current Vocabulary Size.
*   **$L$**: Length of the pattern.

---

## ⚔️ Comparative Analysis

Where does **TreeMemoryPredictor** fit in the Machine Learning landscape?

| Feature | N-Gram / T9 | Neural Networks (LSTM/GPT) | PPM (Compression) | **TreeMemoryPredictor (TMP)** |
| :--- | :--- | :--- | :--- | :--- |
| **Context Window** | Fixed (e.g., last 2 words) | Long / Attention-based | Variable (Unlimited) | **Variable (1..N)** |
| **Training Time** | None | Hours to Months (GPU) | None | **None (Instant)** |
| **Adaptability** | Low (Static Dictionary) | Low (Frozen Weights) | High | **Very High (Lazy Decay)** |
| **Recall** | Frequency-based | Semantic / Fuzzy | Exact Match | **Exact Match + Decay** |
| **Hardware** | Calculator-tier | GPU Cluster | CPU | **CPU** |

---

## 🛠 Configuration

| Parameter | Type | Default | Optimal Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | `3` - `50` | Maximum context length (Markov order). Higher values capture longer dependencies. |
| `n_min` | `int` | `1` | `1` - `3` | Minimum context length to consider during prediction. Set >1 to ignore single-token noise. |
| `decay` | `float` | `0.99` | `0.90` - `0.999` | Memory retention rate. `0.90` adapts very fast; `0.999` keeps long-term memory stable. |
| `alphabet_autoscale` | `bool` | `True` | `True` | **Highly Recommended.** Scales weights by $VocabSize^{Length}$. |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you have ideas for optimizing the Trie structure or adding new weighting schemes.

## 📄 License

This project is licensed under the MIT License.