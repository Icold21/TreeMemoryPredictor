# TreeMemoryPredictor (TMP) 🧠

A high-performance, adaptive sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**. Implemented in pure Python with strict typing and zero external dependencies.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** learns patterns "on the fly." Unlike deep learning models, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to new data streams while smoothly forgetting outdated information.

### Why choose TMP over a Neural Network?
* **Instant Learning:** Calling `model.update(token)` updates probabilities in strictly $O(1)$ amortized time. The next prediction is immediately smarter.
* **Privacy-First & Edge Ready:** Runs entirely locally. Perfect for on-device personalization (e.g., custom keyboards predicting a user's unique slang).
* **Asymmetric Federated Learning:** Natively merges knowledge from heterogeneous distributed models, adaptively expanding its context capabilities while respecting individual mathematical decays. 
* **Explainable Math:** No black boxes. Every probability is mathematically derived from exact historical occurrences and explicit decay formulas.

---

## ✨ Key Features & Architecture

1. **Reverse Suffix Trie ($O(N)$ Traversal):** Context is processed backwards. This architectural choice drops sequence lookup complexity from $O(N^2)$ down to strictly $O(N)$.
2. **Lazy Exponential Decay:** Implements $O(1)$ weight updates relative to history length. Old patterns smoothly fade away ($Count \times Decay^{\Delta t}$), allowing the model to adapt dynamically to changing trends.
3. **Asymmetric Federated Merging:** Built-in `merge()` method allows distributed clients to combine their learned state into a global Super-Model. The target model dynamically expands its depth limits (`max(n_max)`) and accurately calculates the present value of foreign knowledge using the client's original decay rate.
4. **Katz-style Backoff Smoothing:** Automatically gracefully degrades to $O(1)$ tracked unigram frequencies when encountering entirely unseen contexts, replacing naive uniform guesses with historically accurate fallback distributions.
5. **Adaptive Garbage Collection:** Features a dynamic Memory Management engine. Prunes dead branches based on true-decay math either at fixed intervals or adaptively tracking tree growth, ensuring bounded RAM usage over infinite data streams.
6. **Context Masking via Beam Search:** Optional wildcard evaluation for noisy data (`masked_mode`). Utilizes bounded Breadth-First Search (`max_beams`) to identify probabilistic correlations even across typos, preventing combinatorial explosions.

---

## 🚀 Quick Start

### Installation
Simply copy the `tmp.py` class code into your project or repository. *Strictly supports `str` and `int` tokens.*

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

TMP provides familiar parameters to control generation randomness and confidence.

```python
# Temperature > 1.0 makes it creative, < 1.0 makes it strict and confident
next_token = model.predict(temperature=1.2, top_k=5)

# Nucleus Sampling (Dynamically limits candidates to top 90% of probability mass)
next_token = model.predict(top_p=0.9)
```

### 3. Federated Learning (Heterogeneous Merging)

Combine knowledge from distributed instances (e.g., models trained locally on different mobile devices). You can seamlessly merge models with different parameters (e.g., deeper memory or faster decay). The core engine mathematically syncs timelines, adjusts bounds, and scales knowledge safely.

```python
# A lightweight fast-forgetting global baseline
global_model = TreeMemoryPredictor(n_max=3, n_min=2, decay=0.9)

# An advanced edge device with deep memory and long-term retention
client_a = TreeMemoryPredictor(n_max=10, n_min=1, decay=0.999).fit(['deep', 'logic', 'path'])

# Merge local insights: The Global model will adaptively upgrade its capabilities
# to n_max=10, n_min=1 while absorbing client_a's exact weighted knowledge.
global_model.merge(client_a)
```

### 4. Generative Use Case: Character-Level Text

You can feed TMP raw characters to create an instant, lightweight text generator.

```python
text_data = "to be, or not to be, that is the question"

# n_max=12 handles character-level context easily
text_model = TreeMemoryPredictor(n_max=12, decay=0.999)
text_model.fit(text_data)

# Generate novel text
text_model.fill_context(list("to be, "))
generated =[]

for _ in range(30):
    char = text_model.predict(temperature=0.7, top_p=0.95)
    generated.append(char)
    text_model.update(char)

print("".join(generated))
```

### 5. Handling Noisy Data (Masked Inference)

Real-world data contains noise. If a user inputs `["open", "red", "door"]` instead of `["open", "wooden", "door"]`, strict matching fails. TMP bridges this gap:

* **`none`:** Strict exact sequence match (Default & Fastest).
* **`linear`:** Ignores recent noisy tokens (treats them as wildcards), but strictly matches older history.
* **`squared`:** Full combinatorial beam search. Any token can independently be matched or ignored.

```python
# Enable bounded wildcard fallback to find correlations across noisy sequences
model.predict(masked_mode='linear')
```

---

## 📊 Performance Benchmarks

*Hardware: Single Core, Apple M1 (Python 3.10)*

| Operation | Throughput (Tokens/sec) | Memory Overhead | Complexity |
| :--- | :--- | :--- | :--- |
| **Stream Updating (`update`)** | ~125,000 iters/sec | $O(N_{max})$ nodes | $O(N_{max})$ |
| **Inference (`predict_proba`)** | ~85,000 iters/sec | $O(1)$ per step | $O(N_{max} + V_{ctx})$ |
| **Federated Merge** | N/A | $O(Nodes_{foreign})$ | Bounded by Tree Size |
| **Memory Constraint (1M steps)** | N/A | ~45 MB max | Bounded by GC |

---

## 🛠 Configuration Guide

### Initialization Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | Maximum sequence length to remember. `3-5` for words, `10-20` for characters. |
| `n_min` | `int` | `1` | Minimum effective context length required to accept a match. |
| `decay` | `float` | `0.99` | Memory retention rate. `0.9` adapts fast to new trends; `0.999` keeps long-term memory. |
| `fallback_mode` | `str` | `'katz_backoff'`| Smoothing fallback: `'katz_backoff'` (Unigram frequencies) or `'uniform'`. |
| `pruning_mode` | `str` | `'fixed'` | `'fixed'` prunes by steps. `'dynamic'` triggers adaptive GC tracking tree density. |
| `pruning_step` | `int` | `1000` | GC threshold (Interval in `fixed` mode, or starting threshold in `dynamic` mode). |
| `pruning_threshold` | `float` | `1e-6` | Minimum weight threshold. Patterns decaying below this value are permanently deleted. |
| `max_beams` | `int` | `1000` | Prevents RAM/CPU explosion by limiting path exploration in `masked_mode`. |

### Prediction Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `temperature` | `float` | `1.0` | Flattens (`>1.0`) or sharpens (`<1.0`) the distribution. |
| `top_k` | `int` | `0` | Hard cutoff. Keeps only the $K$ most likely tokens. `0` disables it. |
| `top_p` | `float` | `1.0` | Nucleus sampling cutoff. Drops the long tail of low-probability tokens. |
| `masked_mode` | `str` | `'none'` | Search strategy: `'none'`, `'linear'`, or `'squared'`. |

---

## 🤝 Contributing
Contributions, issue reports, and pull requests are warmly welcomed! If you have ideas for adding C-extensions or further metric integrations, feel free to open a PR.

## 📄 License
This project is licensed under the **MIT License**.