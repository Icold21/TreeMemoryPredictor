TreeMemoryPredictor (TMP) 🧠

A high-performance, adaptive sequence prediction engine based on Context Mixing and Variable Order Markov Models (VOMM). Implemented in pure Python with zero external dependencies.

![alt text](https://img.shields.io/badge/Python-3.7%2B-blue)


![alt text](https://img.shields.io/badge/License-MIT-green)


![alt text](https://img.shields.io/badge/Dependencies-Zero-lightgrey)

TreeMemoryPredictor learns patterns "on the fly." Unlike deep learning models, it requires no training epochs, has zero cold-start latency, and adapts instantly to new data streams while smoothly forgetting outdated information.

Why choose TMP over a Neural Network?

Instant Learning: Calling model.update(token) updates probabilities in 
𝑂
(
𝑁
)
O(N)
 time. The next prediction is immediately smarter.

Privacy-First: Runs entirely locally. Perfect for on-device personalization (e.g., a keyboard predicting a user's unique slang).

Explainable Math: No black box. Every probability is mathematically derived from exact historical occurrences and decay formulas.

✨ Key Features & Architecture

Reverse Suffix Trie (
𝑂
(
𝑁
)
O(N)
 Traversal): Context is processed backwards (from the most recent token to the oldest). This evolutionary architecture drops sequence lookup complexity from 
𝑂
(
𝑁
2
)
O(N
2
)
 down to strictly 
𝑂
(
𝑁
)
O(N)
.

Skip-Grams & Masked Sequences (masked_mode): Native support for wildcard matching. Finds probabilistic correlations even if random "noise" or typos are inserted between significant patterns.

Lazy Exponential Decay: Implements 
𝑂
(
1
)
O(1)
 weight updates relative to history length. Old patterns smoothly fade away (
𝐶
𝑜
𝑢
𝑛
𝑡
×
𝐷
𝑒
𝑐
𝑎
𝑦
Δ
𝑡
Count×Decay
Δt
).

True-Decay Garbage Collection: An automated, memory-bounding pruning system that applies exact time-decay formulas before wiping dead branches. Prevents memory leaks over infinite data streams.

Micro-Optimized pure Python: Uses __slots__, lazy-populated math caches, and the Log-Sum-Exp trick to prevent numerical underflow, running as fast as mathematically possible in Python.

🎭 The Magic of masked_mode (Skip-Grams)

Real-world data is noisy. If your model learned ["open", "wooden", "door"], but the user inputs ["open", "red", "door"], a standard Markov chain fails because the exact sequence is broken.

TMP solves this using Breadth-First Search (BFS) wildcard evaluation.

💡 Core Concept: The modes are cumulative. Advanced modes automatically include the standard exact-match search, simply adding more wildcard fallback paths!

Imagine the model learned the phrase: A -> B -> C -> Target.
Your current input context is [A, B, X] (where X is an unexpected typo).

none (Strict Match):
Standard Markov behavior. Looks only for exact unbroken sequences.
Looks for: [X], [B, X], [A, B, X] ❌ (Fails to predict Target)

linear (Recent-Noise Filter):
Includes none, PLUS allows skipping the most recent tokens (treating them as wildcards *), but demands a strict match for the older tokens.
Looks for: [*, *], [B, *], [A, B, *] ✅ (Matches A, B and successfully predicts Target!)

squared (Full Combinatorial):
Includes none and linear, PLUS allows any token in the context to independently be a mask or an exact match.
Can handle middle-noise like [A, *, C] or [*, B, C].

🚀 Quick Start
Installation

Simply copy the TreeMemoryPredictor class code into your project.

1. Basic Usage (Continuous Stream)
code
Python
download
content_copy
expand_less
from tmp import TreeMemoryPredictor

# Initialize: Max context of 5, Forgets at a rate of 0.9 per step
model = TreeMemoryPredictor(n_max=5, decay=0.9)

sequence = ['click', 'buy', 'click', 'buy', 'click']

# Online Learning loop
for token in sequence:
    # 1. Predict next token (Greedy / Argmax)
    prediction = model.predict() 
    print(f"Predicted next step: {prediction}")
    
    # 2. Ingest the actual token & instantly learn it
    model.update(token)
2. Skip-Grams and Masked Inference
code
Python
download
content_copy
expand_less
# Standard search (Fastest, Exact match only)
model.predict(masked_mode='none')

# Linear Skip-Gram (Filters out recent noise/typos)
model.predict(masked_mode='linear')

# Squared (Exhaustive wildcard search for complex correlations)
model.predict(masked_mode='squared')
3. Advanced Generation (Like an LLM)

You can control the randomness of the prediction using standard LLM parameters.

code
Python
download
content_copy
expand_less
# Temperature > 1.0 makes it creative, < 1.0 makes it strict and confident
next_token = model.predict(temperature=1.2, top_k=5)

# Nucleus Sampling (Dynamically limits candidates to top 90% of probability mass)
next_token = model.predict(top_p=0.9)
4. Batch Training & Context Management

Handle independent sequences (e.g., separate user sessions or completely different documents) by clearing the context between them.

code
Python
download
content_copy
expand_less
dataset = [['user', 'login', 'view_item', 'logout'],
    ['admin', 'login', 'delete_user', 'logout']
]

# model.fit() automatically flushes context between sequences
model.fit(dataset) 

# Inference: Manually set context to simulate an active session
model.fill_context(['admin', 'login'])
probs = model.predict_proba() 
# -> {'delete_user': 0.9, 'logout': 0.1}
5. Saving and Loading

TMP handles dynamic internal caches safely during serialization.

code
Python
download
content_copy
expand_less
# Save the model state to disk
model.save("my_predictor.pkl")

# Load it back later
restored_model = TreeMemoryPredictor.load("my_predictor.pkl")
🔬 Under the Hood: The Math

The algorithm uses a weighted voting system across multiple Markov orders (Prediction by Partial Matching).

The Weighting Formula (Log-Space)

Each valid node in the Trie calculates its "voting power" using this exact formula:

log
⁡
(
𝑊
𝑒
𝑖
𝑔
ℎ
𝑡
)
=
log
⁡
(
𝑁
𝑜
𝑏
𝑠
)
+
Δ
𝑡
⋅
log
⁡
(
𝐷
𝑒
𝑐
𝑎
𝑦
)
+
𝐿
𝑒
𝑓
𝑓
⋅
log
⁡
(
𝑆
)
log(Weight)=log(N
obs
	​

)+Δt⋅log(Decay)+L
eff
	​

⋅log(S)

Where:

𝑁
𝑜
𝑏
𝑠
N
obs
	​

: Frequency count of the token.

𝐷
𝑒
𝑐
𝑎
𝑦
Decay
: Forgetting factor (e.g., 0.99).

Δ
𝑡
Δt
: Time steps since this node was last updated.

𝐿
𝑒
𝑓
𝑓
L
eff
	​

: Effective matched context length (Ignores wildcard * positions).

𝑆
S
: Current Vocabulary Size (Dynamically scales the reward for matching long patterns based on alphabet complexity).

🛠 Configuration Guide
Initialization Parameters
Parameter	Type	Default	Description
n_max	int	10	Maximum sequence length to remember. 3-5 for words, 10-20 for characters.
n_min	int	1	Minimum effective context length required to accept a match.
decay	float	0.99	Memory retention rate. 0.9 adapts fast to new trends; 0.999 keeps long-term memory.
alphabet_autoscale	bool	True	Balances pattern-matching entropy. Keep this True in almost all cases.
pruning_step	int	1000	How often (in steps) to execute the Garbage Collector.
cache_size	int	4096	Size of internal lazy math caches. Increase if running on massive datasets with slow decay.
Prediction Parameters (predict / predict_proba)
Parameter	Type	Default	Description
temperature	float	1.0	Flattens (>1.0) or sharpens (<1.0) the distribution.
top_k	int	0	Hard cutoff. Keeps only the 
𝐾
K
 most likely tokens. 0 disables it.
top_p	float	1.0	Nucleus sampling cutoff. Drops the long tail of low-probability tokens.
masked_mode	str	'none'	Search strategy: 'none', 'linear', or 'squared'.
🤝 Contributing

Contributions, issue reports, and pull requests are warmly welcomed! If you have ideas for adding new traversal mechanics, pruning optimizations, or Kneser-Ney smoothing integrations, feel free to open a PR.

📄 License

This project is licensed under the MIT License.