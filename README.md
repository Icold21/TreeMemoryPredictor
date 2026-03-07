```markdown
# TreeMemoryPredictor (TMP) 🧠

A high-performance, adaptive sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**. Implemented in pure Python with zero external dependencies.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** learns patterns "on the fly." Unlike Neural Networks, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to non-stationary data distributions via a **Lazy Decay** mechanism.

It is particularly effective for:
*   **Code Autocompletion:** Local, privacy-first IntelliSense that adapts to your current file's variable names.
*   **Stream Processing:** Anomaly detection in IoT sensors or logs.
*   **Game AI:** Real-time opponent modeling.
*   **Data Compression:** PPM-style probability estimation.

---

## ✨ Key Features & Optimizations

1.  **Dynamic Suffix Trie:** Builds a variable-length context tree (from $N_{min}$ to $N_{max}$) in real-time.
2.  **Lazy Exponential Decay:** Implements $O(1)$ weight updates relative to history length. Old patterns fade away ($Count \times Decay^{\Delta t}$), allowing the model to adapt to changing data.
3.  **Advanced Sampling:** Supports **Temperature**, **Top-K**, and **Nucleus (Top-P)** sampling for generation control.
4.  **Memory Efficient:**
    *   **NBuffer:** Custom circular buffer implementation to avoid $O(N)$ list overheads in Python.
    *   **Periodic Pruning:** Automatic garbage collection of dead branches to bound memory usage.
    *   **__slots__:** Minimal RAM footprint per node.
5.  **Math Cache:** Pre-computes logarithms and decay powers to speed up the hottest loops.
6.  **Log-Space Arithmetic:** All calculations are performed in log-space to prevent numerical underflow.

---

## 🚀 Quick Start

### Installation
Simply copy the class code into your project. There are no heavy dependencies.

### 1. Basic Usage (Continuous Stream)

```python
from tmp import TreeMemoryPredictor

# Initialize: Context up to 5 tokens, Fast adaptation (Decay 0.9)
model = TreeMemoryPredictor(n_max=5, decay=0.9)

sequence = ['click', 'buy', 'click', 'buy', 'click']

# Online Learning
for token in sequence:
    # Predict next token (Greedy / Argmax)
    prediction = model.predict() 
    print(f"Predicted: {prediction}")
    
    # Learn & Move window
    model.update(token) 
```

### 2. Advanced Sampling (Generation)

You can control the randomness of the prediction, similar to LLMs.

```python
# Temperature > 1.0 (Creative), Top-K=5 (Limit candidates)
next_token = model.predict(temperature=1.2, top_k=5)

# Nucleus Sampling (Top-P = 0.9)
next_token = model.predict(top_p=0.9)
```

### 3. Batch Training & Context Management

Handle independent sequences (e.g., separate user sessions or files) by clearing context between them.

```python
dataset = [
    ['user', 'login', 'logout'],
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

The algorithm uses a **weighted voting system** across multiple Markov orders (Prediction by Partial Matching).

### The Weighting Formula (Log-Space)
Each node in the Trie calculates its "voting power" using this formula:

$$
\log(Weight) = \log(N_{obs}) + \Delta t \cdot \log(Decay) + L \cdot \log(S)
$$

Where:
*   **$N_{obs}$**: Frequency count.
*   **$Decay$**: Forgetting factor (e.g., 0.99).
*   **$\Delta t$**: Time steps since this node was last visited.
*   **$S$**: Current Vocabulary Size (Entropy Scaling).
*   **$L$**: Context Length (Order).

---

## 🛠 Configuration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | Maximum context length. For words, `3-5` is usually sufficient. For characters, `10-20`. |
| `n_min` | `int` | `1` | Minimum context length. Set >1 to ignore single-token noise. |
| `decay` | `float` | `0.99` | Memory retention. `0.9` adapts fast; `0.999` keeps long-term memory. |
| `alphabet_autoscale` | `bool` | `True` | **Recommended.** Scales weights by $\log(VocabSize)$ to balance entropy. |
| `pruning_step` | `int` | `1000` | How often (in steps) to run garbage collection to free memory. |
| `cache_size` | `int` | `4096` | Size of internal math caches for logs/powers. |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you have ideas for optimizing the Trie structure or adding new weighting schemes.

## 📄 License

This project is licensed under the MIT License.