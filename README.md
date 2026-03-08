# TreeMemoryPredictor (TMP) 🧠

A high-performance, adaptive sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**. Implemented in pure Python with zero external dependencies.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** learns patterns "on the fly." Unlike Neural Networks, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to non-stationary data distributions via a mathematically exact **Lazy Decay** mechanism.

It is particularly effective for:
*   **Code Autocompletion:** Local, privacy-first IntelliSense that adapts to your current file's variable names.
*   **Stream Processing:** Anomaly detection in IoT sensors or logs.
*   **Game AI:** Real-time opponent modeling.
*   **Data Compression:** PPM-style probability estimation.

---

## ✨ Key Features & Optimizations

1.  **Reverse Suffix Trie ($O(N)$ Traversal):** Context is processed backwards (from the most recent token to the oldest). This revolutionary change drops sequence lookup complexity from $O(N^2)$ down to strictly $O(N)$, drastically speeding up inference and updates.
2.  **Skip-Grams & Masked Sequences:** Native support for wildcard matching (`masked_mode`). The model can find probabilistic correlations even if random "noise" tokens are inserted between significant patterns.
3.  **Lazy Exponential Decay:** Implements $O(1)$ weight updates relative to history length. Old patterns fade away ($Count \times Decay^{\Delta t}$), allowing the model to adapt to changing data continuously.
4.  **True-Decay Garbage Collection:** An automated, memory-bounding pruning system that applies exact time-decay formulas *before* wiping dead branches, guaranteeing zero memory leaks over infinite streams.
5.  **Advanced Sampling:** Supports **Temperature**, **Top-K**, and highly optimized CPU-friendly **Nucleus (Top-P)** sampling.
6.  **Micro-Optimized Python:**
    *   **NBuffer:** Custom stateful circular buffer bypassing $O(N)$ operations for size checks.
    *   **Lazy Math Caches:** Pre-computes logarithms and decay powers *on-the-fly* up to a strict capacity limit.
    *   **__slots__ & BFS Caching:** Minimal RAM footprint per node and C-level dictionary ID lookups.
    *   **Log-Space Arithmetic:** All calculations use the Log-Sum-Exp trick to prevent numerical underflow on deep n-grams.

---

## 🚀 Quick Start

### Installation
Simply copy the class code into your project. There are no heavy dependencies.

### 1. Basic Usage (Continuous Stream)

```python
from tmp import TreeMemoryPredictor

# Initialize: Context up to 5 tokens, Fast adaptation (Decay 0.9)
model = TreeMemoryPredictor(n_max=5, decay=0.9)

sequence =['click', 'buy', 'click', 'buy', 'click']

# Online Learning
for token in sequence:
    # Predict next token (Greedy / Argmax)
    prediction = model.predict() 
    print(f"Predicted: {prediction}")
    
    # Learn & Move window
    model.update(token) 
```

### 2. Advanced Sampling (Generation)

You can control the randomness of the prediction, similar to modern LLMs.

```python
# Temperature > 1.0 (Creative), Top-K=5 (Limit candidates)
next_token = model.predict(temperature=1.2, top_k=5)

# Nucleus Sampling (Top-P = 0.9)
next_token = model.predict(top_p=0.9)
```

### 3. Skip-Grams and Masked Inference 🆕

TMP can predict the next token even if the recent context is noisy, by treating certain positions in the n-gram as wildcards.

```python
# 1. Standard search (Exact match only)
model.predict(masked_mode='none')

# 2. Linear Skip-Gram: 
# Ignores recent noisy tokens (acts as a wildcard mask), 
# but strictly matches older historical tokens once a pattern is found.
model.predict(masked_mode='linear')

# 3. Squared (Full Combinatorial): 
# Exhaustive BFS wildcard search. Every single token in the context 
# can independently be a mask or an exact match.
model.predict(masked_mode='squared')
```

### 4. Batch Training & Context Management

Handle independent sequences (e.g., separate user sessions or files) by clearing context between them.

```python
dataset = [['user', 'login', 'logout'],
    ['admin', 'login', 'delete', 'logout']
]

# Fit on batch (Context resets automatically between rows)
model.fit(dataset) 

# Inference with specific context
model.fill_context(['admin', 'login'])
probs = model.predict_proba() 
# -> {'delete': 0.9, 'logout': 0.1}
```

---

## 🔬 How It Works

The algorithm uses a **weighted voting system** across multiple Markov orders (Prediction by Partial Matching) combined with a **Reverse Suffix Trie**.

### The Weighting Formula (Log-Space)
Each valid node in the Trie calculates its "voting power" using this formula:

$$
\log(Weight) = \log(N_{obs}) + \Delta t \cdot \log(Decay) + L_{eff} \cdot \log(S)
$$

Where:
*   **$N_{obs}$**: Frequency count of the token.
*   **$Decay$**: Forgetting factor (e.g., 0.99).
*   **$\Delta t$**: Time steps since this node was last visited.
*   **$L_{eff}$**: Effective matched context length (Important for `masked_mode`).
*   **$S$**: Current Vocabulary Size (Entropy Scaling).

---

## 🛠 Configuration

### Initialization Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | Maximum context length. For words, `3-5` is usually sufficient. For characters, `10-20`. |
| `n_min` | `int` | `1` | Minimum effective context length required to accept a match. |
| `decay` | `float` | `0.99` | Memory retention. `0.9` adapts fast; `0.999` keeps long-term memory. |
| `alphabet_autoscale` | `bool` | `True` | **Recommended.** Scales weights by $\log(VocabSize)$ to balance entropy. |
| `pruning_step` | `int` | `1000` | How often (in steps) to run garbage collection to free memory. |
| `cache_size` | `int` | `4096` | Size of internal lazy math caches for logs/powers. |

### Prediction Parameters (`predict` / `predict_proba`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `temperature` | `float` | `1.0` | Flattens (`>1.0`) or sharpens (`<1.0`) the probability distribution. |
| `top_k` | `int` | `0` | Keeps only the $K$ most likely tokens. `0` disables it. |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold. `1.0` disables it. |
| `masked_mode` | `str` | `'none'` | Context matching strategy: `'none'`, `'linear'`, or `'squared'`. |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you have ideas for optimizing the Trie structure, introducing new masked BFS algorithms, or adding new weighting schemes.

## 📄 License

This project is licensed under the MIT License.