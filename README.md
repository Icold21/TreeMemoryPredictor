# TreeMemoryPredictor (TMP) 🧠

A high-performance, adaptive sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**. Implemented in pure Python with zero external dependencies.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** learns patterns "on the fly." Unlike deep learning models, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to new data streams while smoothly forgetting outdated information.

### Why choose TMP over a Neural Network?
* **Instant Learning:** Calling `model.update(token)` updates probabilities in $O(N)$ time. The next prediction is immediately smarter.
* **Privacy-First:** Runs entirely locally. Perfect for on-device personalization (e.g., a custom keyboard predicting a user's unique slang).
* **Explainable Math:** No black box. Every probability is mathematically derived from exact historical occurrences and explicit decay formulas.

---

## ✨ Key Features & Architecture

1. **Reverse Suffix Trie ($O(N)$ Traversal):** Context is processed backwards (from the most recent token to the oldest). This evolutionary architecture drops sequence lookup complexity from $O(N^2)$ down to strictly $O(N)$.
2. **Lazy Exponential Decay:** Implements $O(1)$ weight updates relative to history length. Old patterns smoothly fade away ($Count \times Decay^{\Delta t}$), allowing the model to adapt to changing trends.
3. **True-Decay Garbage Collection:** An automated, memory-bounding pruning system that applies exact time-decay formulas *before* wiping dead branches. Prevents memory leaks over infinite data streams.
4. **Context Masking (`masked_mode`):** Optional wildcard evaluation via BFS. Allows the model to find probabilistic correlations even if random "noise" or typos are inserted between significant patterns.
5. **Micro-Optimized pure Python:** Uses `__slots__`, lazy-populated math caches, and the Log-Sum-Exp trick to prevent numerical underflow, running as fast as mathematically possible in Python.

---

## 🚀 Quick Start

### Installation
Simply copy the `tmp.py` class code into your project.

### 1. Basic Usage (Continuous Stream)

```python
from tmp import TreeMemoryPredictor

# Initialize: Max context of 5, Forgets at a rate of 0.9 per step
model = TreeMemoryPredictor(n_max=5, decay=0.9)

sequence =['click', 'buy', 'click', 'buy', 'click']

# Online Learning loop
for token in sequence:
    # 1. Predict next token (Greedy / Argmax)
    prediction = model.predict() 
    print(f"Predicted next step: {prediction}")
    
    # 2. Ingest the actual token & instantly learn it
    model.update(token) 
```

### 2. Advanced Generation (LLM-style Sampling)

You can control the randomness of the prediction using standard LLM parameters.

```python
# Temperature > 1.0 makes it creative, < 1.0 makes it strict and confident
next_token = model.predict(temperature=1.2, top_k=5)

# Nucleus Sampling (Dynamically limits candidates to top 90% of probability mass)
next_token = model.predict(top_p=0.9)
```

### 3. Handling Noisy Data (Masked Inference)

Real-world data can contain noise or typos (e.g., `["open", "red", "door"]` instead of `["open", "wooden", "door"]`). TMP can fallback to wildcard matching to bridge the gap:

* **`none`:** Strict exact sequence match (Default & Fastest).
* **`linear`:** Ignores recent noisy tokens (treats them as wildcards), but strictly matches older history.
* **`squared`:** Full combinatorial search. Any token in the context can independently be matched or ignored.

```python
# Enable wildcard fallback to find correlations across noisy sequences
model.predict(masked_mode='linear')
```

### 4. Batch Training & Context Management

Handle independent sequences (e.g., separate user sessions or completely different documents) by clearing the context between them.

```python
dataset = [
    ['user', 'login', 'view_item', 'logout'],
    ['admin', 'login', 'delete_user', 'logout']
]

# model.fit() automatically flushes context between sequences
model.fit(dataset) 

# Inference: Manually set context to simulate an active session
model.fill_context(['admin', 'login'])
probs = model.predict_proba() 
# -> {'delete_user': 0.9, 'logout': 0.1}
```

### 5. Saving and Loading

TMP handles dynamic internal caches safely during serialization.

```python
# Save the model state to disk
model.save("my_predictor.pkl")

# Load it back later
restored_model = TreeMemoryPredictor.load("my_predictor.pkl")
```

---

## 🔬 Under the Hood: The Math

The algorithm uses a **weighted voting system** across multiple Markov orders (Prediction by Partial Matching).

### The Weighting Formula (Log-Space)
Each valid node in the Trie calculates its "voting power" using this exact formula:

$$
\log(Weight) = \log(N_{obs}) + \Delta t \cdot \log(Decay) + L_{eff} \cdot \log(S)
$$

Where:
* **$N_{obs}$**: Frequency count of the token.
* **$Decay$**: Forgetting factor (e.g., 0.99).
* **$\Delta t$**: Time steps since this node was last updated.
* **$L_{eff}$**: Effective matched context length (Ignores wildcard `*` positions).
* **$S$**: Current Vocabulary Size (Dynamically scales the reward for matching long patterns based on alphabet complexity).

---

## 🛠 Configuration Guide

### Initialization Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | Maximum sequence length to remember. `3-5` for words, `10-20` for characters. |
| `n_min` | `int` | `1` | Minimum effective context length required to accept a match. |
| `decay` | `float` | `0.99` | Memory retention rate. `0.9` adapts fast to new trends; `0.999` keeps long-term memory. |
| `alphabet_autoscale` | `bool` | `True` | Balances pattern-matching entropy. Keep this `True` in almost all cases. |
| `pruning_step` | `int` | `1000` | How often (in steps) to execute the Garbage Collector. |
| `cache_size` | `int` | `4096` | Size of internal lazy math caches. Increase if running on massive datasets with slow decay. |

### Prediction Parameters (`predict` / `predict_proba`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `temperature` | `float` | `1.0` | Flattens (`>1.0`) or sharpens (`<1.0`) the distribution. |
| `top_k` | `int` | `0` | Hard cutoff. Keeps only the $K$ most likely tokens. `0` disables it. |
| `top_p` | `float` | `1.0` | Nucleus sampling cutoff. Drops the long tail of low-probability tokens. |
| `masked_mode` | `str` | `'none'` | Search strategy: `'none'`, `'linear'`, or `'squared'`. |

---

## 🤝 Contributing
Contributions, issue reports, and pull requests are warmly welcomed! If you have ideas for adding new traversal mechanics, pruning optimizations, or Kneser-Ney smoothing integrations, feel free to open a PR.

## 📄 License
This project is licensed under the **MIT License**.