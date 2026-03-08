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
    
    Wraps `collections.deque` and maintains a manual size counter to bypass 
    O(N) operations when continuously checking size or converting to tuples.
    """
    __slots__ =['_maxlen', '_deque', '_cache_tuple', '_size']

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cache_tuple = None
        self._size = 0  

    def append(self, item: Any):
        """Appends an item, invalidates the cache, and tracks size in O(1)."""
        self._deque.append(item)
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Any]):
        """Extends the buffer with multiple items and syncs the size."""
        self._deque.extend(items)
        self._cache_tuple = None
        self._size = len(self._deque)

    def clear(self):
        """Clears the buffer and resets trackers."""
        self._deque.clear()
        self._cache_tuple = None
        self._size = 0

    @property
    def size(self) -> int:
        """Returns the current size of the buffer in O(1)."""
        return self._size

    def to_tuple(self) -> Tuple[Any, ...]:
        """Returns an immutable tuple representation of the buffer (Cached)."""
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
    Lightweight Node for the Suffix Trie structure.
    Uses __slots__ to significantly reduce memory footprint.
    """
    __slots__ =['counts', 'children', 'last_visit_step']
    
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
    Variable-order Markov Model utilizing a Reverse Suffix Trie.
    
    Features:
    - O(N) Traversal: Looks backward from the most recent token for optimal updates.
    - Lazy Decay: Weights decay mathematically only upon node visitation.
    - Log-Space Math: Prevents floating-point underflow on deep n-grams.
    - Skip-Grams / Masking: Supports wildcard sequence matching ('linear', 'squared').
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
            n_max: Maximum context order (n-gram length).
            n_min: Minimum context order to consider valid.
            decay: Forgetting factor for old observations (0.0 to 1.0).
            alphabet_autoscale: Adjust log-space probability scaling dynamically.
            pruning_step: Number of updates before Garbage Collection triggers.
            cache_size: Maximum memory bounds for mathematical lazy caches.
        """
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        self.pruning_step = pruning_step
        self.cache_size = cache_size
        
        # Internal counters and trackers
        self._vocab_len = 0 
        self._cached_log_base = 0.69314718056  # Initialized to log(2)
        self._last_computed_vocab_len = 0
        
        # Lazy caches to avoid redundant math operations
        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')

        self._int_log_cache = {}
        self._log_cache_len = 0
        
        self.reset()

    def reset(self):
        """Resets the model to its initial empty state."""
        self.root = TreeMemoryNode()
        self.buffer = NBuffer(maxlen=self.n_max) 
        self.step = 0
        self.known_vocabulary = set()
        self._vocab_len = 0
        
        if hasattr(self, '_power_cache'):
            self._power_cache.clear()
            self._power_cache_len = 0
            
        if hasattr(self, '_int_log_cache'):
            self._int_log_cache.clear()
            self._log_cache_len = 0
            
        return self
    
    @property
    def log_scaling_base(self) -> float:
        """Computes or retrieves the dynamic scaling factor based on vocab size."""
        if not self.alphabet_autoscale:
            return 0.69314718056
        
        # Recalculate only if vocabulary size changed
        if self._vocab_len != self._last_computed_vocab_len:
             self._last_computed_vocab_len = self._vocab_len
             self._cached_log_base = math.log(max(2, self._vocab_len))
             
        return self._cached_log_base

    def _get_decay_factor(self, delta: int) -> float:
        """Lazy calculation and caching of exponential decay powers."""
        if self.decay <= 0: 
            return 0.0
        if delta in self._power_cache:
            return self._power_cache[delta]
            
        val = self.decay ** delta
        
        # Populate cache strictly up to defined capacity limits
        if self._power_cache_len < self.cache_size:
            self._power_cache[delta] = val
            self._power_cache_len += 1
            
        return val

    def _get_log_count(self, count: float) -> float:
        """Cached natural logarithm optimized for integer counts."""
        if count <= 1.0: 
            return 0.0 
        
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

    def _prune_recursive(self, node: TreeMemoryNode, current_step: int, threshold: float = 1e-6):
        """
        Garbage Collector step: Recursively applies true decay and removes empty branches.
        Prevents uncontrolled RAM consumption over time.
        """
        delta = current_step - node.last_visit_step
        decay_factor = self._get_decay_factor(delta) if delta > 0 else 1.0
        
        keys_to_remove =[]
        for token, count in node.counts.items():
            real_count = count * decay_factor
            if real_count < threshold:
                keys_to_remove.append(token)
            else:
                node.counts[token] = real_count
                
        for token in keys_to_remove: 
            del node.counts[token]
            
        node.last_visit_step = current_step

        # Collect and purge dead branches recursively
        empty_children =[]
        for token, child in node.children.items():
            self._prune_recursive(child, current_step, threshold)
            if not child.counts and not child.children:
                empty_children.append(token)
                
        for token in empty_children: 
            del node.children[token]

    def prune_tree(self):
        """Triggers a full GC pass on the Trie."""
        self._prune_recursive(self.root, self.step)

    def _get_context_nodes(self, mode: str, reverse_context: Tuple[Any, ...]) -> List[Tuple[TreeMemoryNode, int]]:
        """
        Searches the Suffix Trie based on the requested masking mode.
        Returns a list of tuples containing valid Nodes and their Effective Matches Length.
        """
        max_depth = len(reverse_context)
        if max_depth == 0: 
            return[]
        
        valid_nodes =[]
        
        if mode == 'none':
            # Fast Standard Path: Strictly sequential backward search (O(N))
            node = self.root
            for i in range(max_depth):
                token = reverse_context[i]
                if token not in node.children: 
                    break
                node = node.children[token]
                if i + 1 >= self.n_min:
                    valid_nodes.append((node, i + 1))
                    
        elif mode == 'linear':
            # Skip-Gram Phase 0 (Masking recent) -> Phase 1 (Strict historical match)
            queue = deque([(self.root, 0, 0, 0)]) 
            visited = {}
            while queue:
                node, depth, phase, eff_len = queue.popleft()
                
                # Keep only the longest effective match path for any specific node to avoid duplication
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (node, eff_len)
                        
                if depth == max_depth: 
                    continue
                target_token = reverse_context[depth]
                
                if phase == 0:
                    for t, child in node.children.items():
                        # Option A: Ignore token mismatch (acting as a mask)
                        queue.append((child, depth + 1, 0, eff_len))
                        # Option B: Exact match shifts logic into Strict Phase 1
                        if t == target_token:
                            queue.append((child, depth + 1, 1, eff_len + 1))
                else:
                    # In Phase 1, masking is disabled; we require an unbroken chain
                    if target_token in node.children:
                        queue.append((node.children[target_token], depth + 1, 1, eff_len + 1))
                        
            valid_nodes = list(visited.values())
            
        elif mode == 'squared':
            # Combinatorial search: All tokens can be independently masked or matched
            queue = deque([(self.root, 0, 0)]) 
            visited = {}
            while queue:
                node, depth, eff_len = queue.popleft()
                
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (node, eff_len)
                        
                if depth == max_depth: 
                    continue
                target_token = reverse_context[depth]
                
                for t, child in node.children.items():
                    # Increase effective length strictly upon positive hits
                    match_len = eff_len + 1 if t == target_token else eff_len
                    queue.append((child, depth + 1, match_len))
                    
            valid_nodes = list(visited.values())
            
        return valid_nodes

    def predict_proba(self, 
                      temperature: Optional[float] = 1.0, 
                      top_k: Optional[int] = 0, 
                      top_p: Optional[float] = 1.0,
                      masked_mode: str = 'none') -> Dict[Any, float]:
        """
        Calculates the probability distribution of the next token.
        Supports advanced sampling algorithms and multiple context matching mechanics.
        """
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0

        hist_len = self.buffer.size
        if hist_len == 0: 
            return {}

        candidate_log_scores = defaultdict(lambda: -float('inf'))
        log_scale_base = self.log_scaling_base
        log_decay_val = self.log_decay
        current_step = self.step
        
        # O(1) Slicing for fast Reverse Context retrieval
        max_depth = min(self.n_max, hist_len)
        reverse_context = tuple(reversed(self.buffer.to_tuple()[-max_depth:]))
        
        valid_nodes = self._get_context_nodes(masked_mode, reverse_context)
        found_pattern = False
        
        # Probability Aggregation (Log-Sum-Exp Trick)
        for node, length in valid_nodes:
            delta = current_step - node.last_visit_step
            node_factor = (delta * log_decay_val) + (length * log_scale_base)
            
            for t, count in node.counts.items():
                if count <= 1e-9: continue
                found_pattern = True
                log_weight = self._get_log_count(count) + node_factor
                
                curr = candidate_log_scores[t]
                if curr == -float('inf'):
                    candidate_log_scores[t] = log_weight
                else:
                    # Log-Sum-Exp avoids float underflow when merging branches
                    if curr > log_weight: 
                        candidate_log_scores[t] = curr + math.log1p(math.exp(log_weight - curr))
                    else: 
                        candidate_log_scores[t] = log_weight + math.log1p(math.exp(curr - log_weight))

        # --- Base Fallback ---
        if not found_pattern:
            if self._vocab_len == 0: 
                return {}
            prob = 1.0 / self._vocab_len
            return {tk: prob for tk in self.known_vocabulary}

        # --- Temperature Scaling ---
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
            
        probas = {t: v / total_sum for t, v in linear_scores.items()}
        
        # Early Exit Optimization
        if k <= 0 and p >= 1.0:
            return dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        sorted_items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        
        # --- Top-K Filtering ---
        if 0 < k < len(sorted_items): 
            sorted_items = sorted_items[:k]

        # --- Top-P (Nucleus) Filtering ---
        if p < 1.0:
            current_total_prob = sum(prob for _, prob in sorted_items)
            # Precompute target threshold to avoid inner loop division (CPU optimization)
            target_prob = p * current_total_prob 
            cumulative_prob = 0.0
            for i, (_, prob) in enumerate(sorted_items):
                cumulative_prob += prob
                if cumulative_prob >= target_prob:
                    sorted_items = sorted_items[:i + 1]
                    break

        # Re-Normalize resulting slice
        new_total = sum(prob for _, prob in sorted_items)
        if new_total > 0: 
            return {tk: prob / new_total for tk, prob in sorted_items}
            
        return dict(sorted_items)

    def predict(self, 
                temperature: Optional[float] = 1.0, 
                top_k: Optional[int] = 0, 
                top_p: Optional[float] = 1.0,
                masked_mode: str = 'none') -> Optional[Any]:
        """Samples a single token based on the internal probabilistic distribution."""
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0
        
        # Extreme low temp flattens logic into greedy argmax
        if temp < 1e-4:
            probas = self.predict_proba(temperature=1.0, top_k=k, top_p=p, masked_mode=masked_mode)
            if not probas: return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(temperature=temp, top_k=k, top_p=p, masked_mode=masked_mode)
        if not probas: 
            return None
            
        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    def update(self, actual: Any):
        """
        Ingests a new observation into the Suffix Trie model in O(N_max) operations.
        Applies Lazy Decay algorithm dynamically for active context branches.
        """
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
        
        hist_len = self.buffer.size
        history_tuple = self.buffer.to_tuple()
        node = self.root
        
        # Forward pass down the reversed sequence path
        for i in range(1, min(self.n_max, hist_len) + 1):
            token = history_tuple[-i]
            
            # Fast graph navigation & node construction
            node = node.children.setdefault(token, TreeMemoryNode())
            
            # Lazy Decay specific to currently visited n-gram path
            if node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = self._get_decay_factor(delta)
                    keys_to_remove =[]
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < 1e-5: 
                            keys_to_remove.append(t)
                        else: 
                            node.counts[t] = new_val
                            
                    for t in keys_to_remove: 
                        del node.counts[t]
            
            node.last_visit_step = current_step
            node.counts[actual] += 1.0
            
        self.buffer.append(actual)
        
        # Check Garbage Collection condition
        if self.step % self.pruning_step == 0: 
            self.prune_tree()

    def fit(self, X: Union[Iterable[Any], Iterable[Iterable[Any]]], verbose: bool = True):
        """
        Batch or Sequence streaming entry point. Automatically detects data shapes.
        """
        is_batch = False
        
        # Safe heuristic to determine if X is a batch of sequences or a single stream
        if hasattr(X, '__len__') and len(X) > 0:
            # We assume it's a sequence if the first item is not a pure string
            first_element = next(iter(X))
            if isinstance(first_element, (list, tuple)) or (hasattr(first_element, '__iter__') and not isinstance(first_element, (str, bytes))):
                is_batch = True

        iterator = X
        if verbose and _tqdm:
            iterator = _tqdm(X, total=len(X) if hasattr(X, '__len__') else None, desc="TMP Fitting", unit="seq" if is_batch else "tok")

        if is_batch:
            for sequence in iterator:
                self.buffer.clear()
                for token in sequence: 
                    self.update(token)
        else:
            for token in iterator: 
                self.update(token)
                
        return self

    def update_context(self, token: Any): 
        """Manually push a token to the buffer without learning/updating weights."""
        self.buffer.append(token)
        
    def fill_context(self, context: Iterable[Any]): 
        """Replaces the entire current context buffer."""
        self.buffer.clear()
        self.buffer.extend(context)
        
    def reset_context(self): 
        """Flushes context."""
        self.buffer.clear()
        
    # --- Persistence ---
    def __getstate__(self):
        """Purges dynamic caches strictly before serialization."""
        state = self.__dict__.copy()
        for k in ['_power_cache', '_int_log_cache']: 
            if k in state: 
                del state[k]
        return state

    def __setstate__(self, state):
        """Restores the object and initializes all lazy caches appropriately."""
        self.__dict__.update(state)
        
        # Fallback bindings for older model variants
        if not hasattr(self, 'cache_size'): self.cache_size = 4096
        if not hasattr(self, 'pruning_step'): self.pruning_step = 1000
        
        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')
        self._int_log_cache = {} 
        self._log_cache_len = 0

        # Legacy history migrations
        if 'history' in state and not hasattr(self, 'buffer'):
            self.buffer = NBuffer(self.n_max)
            self.buffer.extend(state['history'])
            if 'history' in self.__dict__: 
                del self.history
            
        if not hasattr(self, '_vocab_len'): 
            self._vocab_len = len(getattr(self, 'known_vocabulary', set()))
            
        if not hasattr(self, '_cached_log_base'):
             self._cached_log_base = math.log(max(2, self._vocab_len))
        
        self._last_computed_vocab_len = 0

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f: 
                pickle.dump(self, f)
        except Exception as e: 
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'TreeMemoryPredictor':
        try:
            with open(filepath, 'rb') as f: 
                return pickle.load(f)
        except Exception as e: 
            print(f"Error loading model: {e}")
            return None