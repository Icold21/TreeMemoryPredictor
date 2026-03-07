import math
import pickle
import random
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any, Iterable, Union, Tuple

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

class NBuffer:
    """
    Optimized stateful buffer for sliding window operations.

    Wraps `collections.deque` to bypass O(N) overhead of standard list operations
    in Python. Maintains a manual size counter and a cached tuple view for
    fast slicing.
    """
    __slots__ = ['_maxlen', '_deque', '_cache_tuple', '_size']

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cache_tuple = None
        self._size = 0  # Manual tracker to avoid len() calls

    def append(self, item: Any):
        """Appends item, invalidates cache, updates size (O(1))."""
        self._deque.append(item)
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Any]):
        """Extends buffer, invalidates cache, syncs size."""
        self._deque.extend(items)
        self._cache_tuple = None
        self._size = self._deque.__len__()

    def clear(self):
        """Resets buffer and counters."""
        self._deque.clear()
        self._cache_tuple = None
        self._size = 0

    @property
    def size(self) -> int:
        """Returns manually tracked size (O(1))."""
        return self._size

    def to_tuple(self) -> Tuple[Any, ...]:
        """Returns immutable buffer view (Cached O(1))."""
        if self._cache_tuple is None:
            self._cache_tuple = tuple(self._deque)
        return self._cache_tuple

    # --- Persistence ---
    def __getstate__(self):
        return {'_maxlen': self._maxlen, '_deque': self._deque, '_size': self._size}

    def __setstate__(self, state):
        self._maxlen = state.get('_maxlen', 10)
        self._deque = state.get('_deque', deque(maxlen=self._maxlen))
        self._size = state.get('_size', len(self._deque))
        self._cache_tuple = None


class TreeMemoryNode:
    """
    Lightweight Trie Node using __slots__ to minimize RAM usage.
    """
    __slots__ = ['counts', 'children', 'last_visit_step']
    
    def __init__(self):
        self.counts = defaultdict(float) 
        self.children = {}
        self.last_visit_step = 0

    def __getstate__(self):
        return {
            'counts': self.counts,
            'children': self.children,
            'last_visit_step': self.last_visit_step
        }

    def __setstate__(self, state):
        self.counts = state.get('counts', defaultdict(float))
        self.children = state.get('children', {})
        self.last_visit_step = state.get('last_visit_step', 0)


class TreeMemoryPredictor:
    """
    Log-Space Context Mixing Model (TMP).

    A streaming variable-order Markov model (Suffix Tree).
    Optimized for speed and memory efficiency in pure Python.

    Key Features:
        - **Lazy Decay:** O(1) weight updates relative to history.
        - **Math Cache:** Pre-computed logs/powers to avoid math overhead.
        - **Log-Space:** Numerical stability for small probabilities.
        - **Pruning:** Periodic garbage collection of dead branches.
    """

    def __init__(self, 
                 n_max: int = 10, 
                 n_min: int = 1, 
                 decay: float = 0.99, 
                 alphabet_autoscale: bool = True,
                 pruning_step: int = 1000,
                 cache_size: int = 4096):
        """
        Args:
            n_max: Maximum context order.
            n_min: Minimum context order.
            decay: Forgetting factor (0.0 - 1.0).
            alphabet_autoscale: Scale weights by log(VocabSize).
            pruning_step: Steps between garbage collection runs.
            cache_size: Max entries for math caches.
        """
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        self.pruning_step = pruning_step
        self.cache_size = cache_size
        
        # Internal State
        self._vocab_len = 0 # Manual vocab counter
        self._cached_log_base = 0.69314718056 # log(2)
        self._last_computed_vocab_len = 0
        
        # Cache 1: Decay Powers (Static array)
        if self.decay > 0:
            self._power_cache = [self.decay ** i for i in range(self.cache_size)]
            self._power_cache_len = len(self._power_cache)
            self.log_decay = math.log(self.decay)
        else:
            self._power_cache = []
            self._power_cache_len = 0
            self.log_decay = -float('inf')

        # Cache 2: Integer Logs (Lazy Dict)
        self._int_log_cache = {}
        self._log_cache_len = 0 # Manual cache counter
        
        self.reset()

    def reset(self):
        """Resets model state."""
        self.root = TreeMemoryNode()
        self.buffer = NBuffer(maxlen=self.n_max) 
        self.step = 0
        self.known_vocabulary = set()
        self._vocab_len = 0
        
        if hasattr(self, '_int_log_cache'):
            self._int_log_cache.clear()
            self._log_cache_len = 0
        return self
    
    @property
    def log_scaling_base(self) -> float:
        """Calculates entropy scaling factor using manual vocab counter."""
        if not self.alphabet_autoscale:
            return 0.69314718056
        
        if self._vocab_len != self._last_computed_vocab_len:
             self._last_computed_vocab_len = self._vocab_len
             self._cached_log_base = math.log(max(2, self._vocab_len))
             
        return self._cached_log_base

    def _get_decay_factor(self, delta: int) -> float:
        """Fast retrieval of decay factor from cache."""
        if delta < self._power_cache_len:
            return self._power_cache[delta]
        return self.decay ** delta

    def _get_log_count(self, count: float) -> float:
        """
        Cached natural log for integers. 
        Uses manual `_log_cache_len` to avoid `len()` overhead.
        """
        if count <= 1.0: return 0.0 
        
        if count.is_integer():
            ix = int(count)
            if ix in self._int_log_cache:
                return self._int_log_cache[ix]
            
            val = math.log(count)
            if self._log_cache_len < self.cache_size:
                self._int_log_cache[ix] = val
                self._log_cache_len += 1
            return val
        
        return math.log(count)

    def _prune_recursive(self, node: TreeMemoryNode, threshold: float = 1e-6):
        """Recursively removes empty nodes and minimal counts (GC)."""
        # Prune Counts
        for token in list(node.counts.keys()):
            if node.counts[token] < threshold:
                del node.counts[token]

        # Prune Children
        for token in list(node.children.keys()):
            child = node.children[token]
            self._prune_recursive(child, threshold)
            if not child.counts and not child.children:
                del node.children[token]

    def prune_tree(self):
        """Triggers full garbage collection."""
        self._prune_recursive(self.root)

    def predict_proba(self, 
                      temperature: Optional[float] = 1.0, 
                      top_k: Optional[int] = 0, 
                      top_p: Optional[float] = 1.0) -> Dict[Any, float]:
        """
        Predicts next token probabilities.

        Pipeline: Context Mixing -> Temp Scaling -> Top-K -> Nucleus (Top-P).
        
        Args:
            temperature: Flatten/Sharpen distribution (None=1.0).
            top_k: Keep K most likely tokens (None=0).
            top_p: Nucleus sampling threshold (None=1.0).
        """
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0

        if self.buffer.size == 0:
            return {}

        candidate_log_scores = defaultdict(lambda: -float('inf'))
        
        # Localize for speed
        log_scale_base = self.log_scaling_base
        log_decay_val = self.log_decay
        current_step = self.step
        
        history_tuple = self.buffer.to_tuple()
        # hist_len is implicitly used via slicing, buffer.size is O(1)
        hist_len = self.buffer.size 
        found_pattern = False
        
        # --- Context Mixing ---
        for length in range(self.n_min, self.n_max + 1):
            if hist_len < length: break
            
            context = history_tuple[-length:]
            node = self.root
            path_exists = True
            
            for token in context:
                if token not in node.children:
                    path_exists = False
                    break
                node = node.children[token]
            
            if path_exists:
                # Log-Space: log(Count) + Delta*log(Decay) + Length*log(Base)
                delta = current_step - node.last_visit_step
                node_factor = (delta * log_decay_val) + (length * log_scale_base)
                
                for token, count in node.counts.items():
                    if count <= 1e-9: continue
                    found_pattern = True
                    log_weight = self._get_log_count(count) + node_factor
                    
                    # Log-Sum-Exp Trick
                    curr = candidate_log_scores[token]
                    if curr == -float('inf'):
                        candidate_log_scores[token] = log_weight
                    else:
                        if curr > log_weight:
                            candidate_log_scores[token] = curr + math.log1p(math.exp(log_weight - curr))
                        else:
                            candidate_log_scores[token] = log_weight + math.log1p(math.exp(curr - log_weight))

        # --- Fallback ---
        if not found_pattern:
            if self._vocab_len == 0: return {}
            prob = 1.0 / self._vocab_len
            return {k: prob for k in self.known_vocabulary}

        # --- Temperature ---
        if temp != 1.0 and temp > 1e-4:
            for t in candidate_log_scores:
                candidate_log_scores[t] /= temp

        # --- Softmax Normalization ---
        max_log = max(candidate_log_scores.values())
        linear_scores = {}
        total_sum = 0.0
        
        for token, log_score in candidate_log_scores.items():
            val = math.exp(log_score - max_log)
            linear_scores[token] = val
            total_sum += val
            
        probas = {t: v/total_sum for t, v in linear_scores.items()}
        
        # Early Exit
        if k <= 0 and p >= 1.0:
            return dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        # --- Filtering ---
        sorted_items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        list_len = len(sorted_items)

        # Top-K
        if k > 0 and k < list_len:
            sorted_items = sorted_items[:k]

        # Top-P
        if p < 1.0:
            current_total_prob = sum(prob for _, prob in sorted_items)
            cumulative_prob = 0.0
            cutoff_index = 0
            for i, (_, prob) in enumerate(sorted_items):
                cumulative_prob += (prob / current_total_prob)
                if cumulative_prob >= p:
                    cutoff_index = i
                    break
            sorted_items = sorted_items[:cutoff_index + 1]

        # Renormalization
        new_total = sum(p for _, p in sorted_items)
        if new_total > 0:
            return {k: p / new_total for k, p in sorted_items}
        
        return dict(sorted_items)

    def predict(self, 
                temperature: Optional[float] = 1.0, 
                top_k: Optional[int] = 0, 
                top_p: Optional[float] = 1.0) -> Optional[Any]:
        """Samples a single token. Returns None if vocab is empty."""
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0
        
        if temp < 1e-4:
            probas = self.predict_proba(temperature=1.0, top_k=k, top_p=p)
            if not probas: return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(temperature=temp, top_k=k, top_p=p)
        if not probas: return None
        
        tokens = list(probas.keys())
        weights = list(probas.values())
        return random.choices(tokens, weights=weights, k=1)[0]

    def update(self, actual: Any):
        """
        Updates model with new observation.
        Applies Lazy Decay to active path and triggers Periodic Pruning.
        """
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
        
        history_tuple = self.buffer.to_tuple()
        hist_len = self.buffer.size
        root = self.root
        
        # Update Trie Path
        for length in range(1, self.n_max + 1):
            if hist_len < length: break
            
            context = history_tuple[-length:]
            node = root
            for token in context:
                if token not in node.children:
                    node.children[token] = TreeMemoryNode()
                node = node.children[token]
            
            # Lazy Decay
            if node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = self._get_decay_factor(delta)
                    keys_to_remove = []
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < 1e-5: keys_to_remove.append(t)
                        else: node.counts[t] = new_val
                    for t in keys_to_remove: del node.counts[t]
            
            node.last_visit_step = current_step
            node.counts[actual] += 1.0
            
        self.buffer.append(actual)

        # Periodic GC
        if self.step % self.pruning_step == 0:
            self.prune_tree()

    def fit(self, X: Union[Iterable[Any], Iterable[Iterable[Any]]], verbose: bool = True):
        """Fits on data (stream or batch)."""
        is_batch = False
        if hasattr(X, '__len__') and len(X) > 0:
            first = next(iter(X))
            if isinstance(first, (list, tuple)) or (hasattr(first, '__iter__') and not isinstance(first, (str, bytes))):
                is_batch = True

        iterator = X
        if verbose and _tqdm:
            total = len(X) if hasattr(X, '__len__') else None
            iterator = _tqdm(X, total=total, desc="TMP Fitting", unit="seq" if is_batch else "tok")

        if is_batch:
            for sequence in iterator:
                self.buffer.clear()
                for token in sequence:
                    self.update(token)
        else:
            for token in iterator:
                self.update(token)
        return self

    # --- Utils ---
    def update_context(self, token: Any):
        self.buffer.append(token)

    def fill_context(self, context: Iterable[Any]):
        self.buffer.clear()
        self.buffer.extend(context)

    def reset_context(self):
        self.buffer.clear()
        
    # --- Persistence ---
    def __getstate__(self):
        """Prepares pickle: removes caches."""
        state = self.__dict__.copy()
        if '_power_cache' in state: del state['_power_cache']
        if '_int_log_cache' in state: del state['_int_log_cache']
        return state

    def __setstate__(self, state):
        """Restores pickle: rebuilds caches."""
        self.__dict__.update(state)
        
        # Defaults
        if not hasattr(self, 'cache_size'): self.cache_size = 4096
        if not hasattr(self, 'pruning_step'): self.pruning_step = 1000
        
        # Rebuild Decay Cache
        if self.decay > 0:
            self._power_cache = [self.decay ** i for i in range(self.cache_size)]
            self._power_cache_len = len(self._power_cache)
            self.log_decay = math.log(self.decay)
        else:
            self._power_cache = []
            self._power_cache_len = 0
            self.log_decay = -float('inf')
            
        # Rebuild Log Cache
        self._int_log_cache = {} 
        self._log_cache_len = 0

        # Migration
        if 'history' in state and not hasattr(self, 'buffer'):
            self.buffer = NBuffer(self.n_max)
            self.buffer.extend(state['history'])
            if 'history' in self.__dict__: del self.history
            
        if not hasattr(self, '_vocab_len'):
            self._vocab_len = len(self.known_vocabulary) if hasattr(self, 'known_vocabulary') else 0
        if not hasattr(self, '_cached_log_base'):
             self._cached_log_base = math.log(max(2, self._vocab_len))
        
        self._last_computed_vocab_len = 0

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f: pickle.dump(self, f)
        except Exception as e: print(f"Error saving: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'TreeMemoryPredictor':
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"Error loading: {e}"); return None