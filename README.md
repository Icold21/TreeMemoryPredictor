# TreeMemoryPredictor 🧠

A lightweight, adaptive, and explainable sequence prediction engine based on **Context Mixing** and **Variable Order Markov Models (VOMM)**.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

**TreeMemoryPredictor** is designed to predict the next token in a sequence by learning patterns "on the fly." Unlike Neural Networks, it requires **no training epochs**, has **zero cold-start latency**, and adapts instantly to changing data distributions.

It is particularly effective for:
*   Stream processing (IoT sensors, user behavior).
*   Game AI (opponent modeling).
*   Compression algorithms (PPM-style logic).
*   RNG quality testing.

---

## ✨ Key Features

1.  **Dynamic Suffix Trie:** Builds a memory structure of variable-length contexts (from $N=1$ to $N_{max}$) in real-time.
2.  **Entropy Scaling ($S^L$):** Automatically adjusts pattern weights based on vocabulary size.
    *   *Binary:* A match of length 3 is weighted by $2^3$.
    *   *Text:* A match of length 3 is weighted by $\approx 30^3$.
    *   *Result:* Long, precise matches in complex alphabets overpower noise automatically.
3.  **Lazy Exponential Decay:** Implements a "forgetting" mechanism ($Count \times Decay^{\Delta t}$). Old, unused patterns fade away, allowing the model to adapt to non-stationary data (concept drift).
4.  **Any Data Type:** Works with integers, strings, characters, or any hashable object.

---

## 🚀 Quick Start

### Installation
Simply copy `tmp.py` (or rename it to `memory_predictor.py`) into your project. There are no heavy dependencies.

### Basic Usage (Binary/Categorical)

```python
from tmp import TreeMemoryPredictor

# Initialize: Look back 6 steps, Decay old patterns by 1% per step
model = TreeMemoryPredictor(n_max=6, decay=0.99)

sequence = [0, 1, 0, 1, 0, 1, 0]

# Online Learning
for token in sequence:
    prediction = model.predict() # Returns 0 or 1
    probabilities = model.predict_proba() # e.g., {0: 0.1, 1: 0.9}
    
    print(f"Input: {token} | Predicted: {prediction} | Confidence: {probabilities.get(prediction, 0):.2f}")
    
    # Update model with actual observation
    model.update(token)
```

### Advanced Usage (Text)

When working with text, the `alphabet_autoscale=True` parameter (on by default) ensures the model understands that guessing a word is much harder than guessing a bit.

```python
text_model = TreeMemoryPredictor(n_max=8, decay=0.95, alphabet_autoscale=True)
sample_text = "hello world hello python"

for char in sample_text:
    pred = text_model.predict()
    text_model.update(char)
```

---

## 🔬 How It Works

The algorithm uses a **weighted voting system** across multiple Markov orders.

### 1. Context Mixing
For a sequence `... A B C`, the model looks up:
*   `C` (Order 1)
*   `B C` (Order 2)
*   `A B C` (Order 3)

### 2. The Weighting Formula
Each node in the Trie calculates its "voting power" using this formula:

$$
Weight = N_{obs} \times (Decay)^{\Delta t} \times (S)^{L}
$$

Where:
*   **$N_{obs}$**: Number of times this pattern was observed.
*   **$Decay$**: Forgetting factor (e.g., 0.99).
*   **$\Delta t$**: Time steps since this node was last visited.
*   **$S$**: Current Vocabulary Size (automatically detected).
*   **$L$**: Length of the pattern.

### 3. Prediction
Probabilities from all context lengths are aggregated and normalized. Due to the $S^L$ term, the longest matching context effectively determines the prediction, while shorter contexts act as a fallback.

---

## 📊 Performance & Visualization

The repository includes a **Jupyter Notebook** (`experiment.ipynb`) demonstrating:

*   **Sine Wave Reconstruction:** Learning periodic integer sequences.
*   **Pattern Switching:** Adapting to rule changes in real-time.
*   **Natural Language:** Predicting text character-by-character.

*Example of Text Prediction Output:*
> <span style="color:green">Arti</span><span style="color:red">f</span><span style="color:green">icial Intel</span><span style="color:red">l</span><span style="color:green">igen</span><span style="color:red">ce</span>...

---

## 🛠 Configuration

All parameters in the `TreeMemoryPredictor` constructor are **optional**. The defaults work well for general-purpose tasks, but you can tune them to fit your specific data dynamics.

| Parameter | Type | Default | Optimal Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `n_max` | `int` | `10` | `3` - `20` | Maximum context length (Markov order). Higher values capture longer dependencies but increase memory usage. |
| `decay` | `float` | `0.99` | `0.90` - `0.999` | Memory retention rate. `0.90` adapts very fast to changes; `0.999` keeps long-term memory stable. |
| `alphabet_autoscale` | `bool` | `True` | `True` | **Highly Recommended.** Scales weights by $VocabSize^{Length}$. Set to `False` only for strict binary tasks if you prefer raw $2^L$ weighting. |

---

## 📚 Theoretical Foundations

This project sits at the intersection of several classic algorithms. While implemented from scratch for stream processing, it shares DNA with:

*   **Prediction by Partial Matching (PPM):** A statistical data compression technique. Like PPM, this model uses a Suffix Trie to find the longest matching context. However, instead of using "escape probabilities" (standard PPM), we use a weighted **Context Mixing** strategy.
*   **Variable Order Markov Models (VOMM):** Unlike a standard Markov Chain that looks back exactly $N$ steps, this model looks back $1, 2, ..., N$ steps simultaneously and aggregates the results based on their confidence.
*   **Exponential Moving Average (EMA):** The "lazy decay" mechanism is effectively an EMA applied to token counts, allowing the system to handle *concept drift* (when the data source changes its behavior over time).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you have ideas for optimizing the Trie structure or adding new weighting schemes.

## 📄 License

This project is licensed under the MIT License.