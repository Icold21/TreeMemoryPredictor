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
    O(N) operations when continuously checking size or converting to tuples 
    in hot loops.
    """
    __slots__ = ['_maxlen', '_deque', '_cache_tuple', '_size']

    def __init__(self, maxlen: int):
        """
        Initializes the buffer.

        Args:
            maxlen (int): The maximum number of items the buffer can hold.
        """
        self._maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cache_tuple = None
        self._size = 0  

    def append(self, item: Any):
        """
        Appends an item, invalidates the tuple cache, and tracks size in O(1).
        
        Args:
            item (Any): The item to add to the buffer.
        """
        self._deque.append(item)
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Any]):
        """
        Extends the buffer with multiple items and synchronizes the size.
        
        Args:
            items (Iterable[Any]): Items to add.
        """
        self._deque.extend(items)
        self._cache_tuple = None
        self._size = len(self._deque)

    def clear(self):
        """Clears the buffer and resets all internal trackers."""
        self._deque.clear()
        self._cache_tuple = None
        self._size = 0

    @property
    def size(self) -> int:
        """
        Returns the current size of the buffer.
        
        Returns:
            int: Number of elements in the buffer (O(1) operation).
        """
        return self._size

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Returns an immutable tuple representation of the buffer.
        Caches the result to avoid redundant O(N) conversions.
        
        Returns:
            Tuple[Any, ...]: The current buffer state.
        """
        if self._cache_tuple is None:
            self._cache_tuple = tuple(self._deque)
        return self._cache_tuple

    # --- Persistence Methods ---
    def __getstate__(self) -> Dict[str, Any]:
        return {'_maxlen': self._maxlen, '_deque': self._deque, '_size': self._size}

    def __setstate__(self, state: Dict[str, Any]):
        self._maxlen = state.get('_maxlen', 10)
        self._deque = state.get('_deque', deque(maxlen=self._maxlen))
        self._size = state.get('_size', len(self._deque))
        self._cache_tuple = None


class TreeMemoryNode:
    """
    Lightweight Node for the Suffix Trie structure.
    Uses __slots__ to significantly reduce the memory footprint per instance.
    """
    __slots__ =['counts', 'children', 'last_visit_step']
    
    def __init__(self):
        self.counts = defaultdict(float) 
        self.children = {}
        self.last_visit_step = 0

    def __getstate__(self) -> Dict[str, Any]:
        return {
            'counts': self.counts,
            'children': self.children,
            'last_visit_step': self.last_visit_step
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.counts = state.get('counts', defaultdict(float))
        self.children = state.get('children', {})
        self.last_visit_step = state.get('last_visit_step', 0)


class TreeMemoryPredictor:
    """
    Variable-order Markov Model utilizing a Reverse Suffix Trie.
    
    Features:
    - O(N) Traversal: Looks backward from the most recent token.
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
        Initializes the sequence predictor.

        Args:
            n_max (int): Maximum context length (n-gram order) to store and evaluate.
            n_min (int): Minimum effective context length required to accept a match.
            decay (float): Forgetting factor (0.0 to 1.0). Determines how fast old observations fade.
            alphabet_autoscale (bool): If True, scales weights by log(VocabSize) to balance entropy.
            pruning_step (int): Number of steps between Garbage Collection cycles.
            cache_size (int): Maximum size for internal lazy math caches.
        """
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        self.pruning_step = pruning_step
        self.cache_size = cache_size
        
        # Internal counters and scaling variables
        self._vocab_len = 0 
        self._cached_log_base = 0.69314718056  # Initialized to log(2)
        self._last_computed_vocab_len = 0
        
        # Lazy caches to avoid redundant heavy math operations
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
        """
        Computes or retrieves the dynamic scaling factor based on vocabulary size.
        This balances the weight of long contexts against the inherent entropy of the alphabet.

        Returns:
            float: Logarithmic scaling base.
        """
        if not self.alphabet_autoscale:
            return 0.69314718056  # log(2) fallback
        
        # Recalculate only if vocabulary size has changed
        if self._vocab_len != self._last_computed_vocab_len:
             self._last_computed_vocab_len = self._vocab_len
             self._cached_log_base = math.log(max(2, self._vocab_len))
             
        return self._cached_log_base

    def _get_decay_factor(self, delta: int) -> float:
        """
        Lazy calculation and caching of exponential decay powers.

        Args:
            delta (int): The number of time steps elapsed.

        Returns:
            float: The decay multiplier (decay ^ delta).
        """
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
        """
        Cached natural logarithm optimized for integer-like counts.

        Args:
            count (float): The token occurrence count.

        Returns:
            float: Natural logarithm of the count.
        """
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
        Prevents uncontrolled RAM consumption over infinite data streams.

        Args:
            node (TreeMemoryNode): The current Trie node.
            current_step (int): The global time step.
            threshold (float): Weight threshold below which nodes are deleted.
        """
        delta = current_step - node.last_visit_step
        decay_factor = self._get_decay_factor(delta) if delta > 0 else 1.0
        
        # Apply true decay before evaluating threshold
        keys_to_remove =[]
        for token, count in node.counts.items():
            real_count = count * decay_factor
            if real_count < threshold:
                keys_to_remove.append(token)
            else:
                node.counts[token] = real_count
                
        for token in keys_to_remove: 
            del node.counts[token]
            
        # Synchronize node time after aging its weights
        node.last_visit_step = current_step

        # Collect and purge empty child branches recursively
        empty_children =[]
        for token, child in node.children.items():
            self._prune_recursive(child, current_step, threshold)
            if not child.counts and not child.children:
                empty_children.append(token)
                
        for token in empty_children: 
            del node.children[token]

    def prune_tree(self):
        """Triggers a full Garbage Collection pass on the Suffix Trie."""
        self._prune_recursive(self.root, self.step)

    def _get_context_nodes(self, mode: str, reverse_context: Tuple[Any, ...]) -> List[Tuple[TreeMemoryNode, int]]:
        """
        Searches the Reverse Suffix Trie based on the requested masking mode.
        Base strict matching ('none') is always executed. Advanced modes append
        masked paths to the baseline results.

        Args:
            mode (str): Evaluation strategy ('none', 'linear', 'squared').
            reverse_context (Tuple): Current buffer history reversed.

        Returns:
            List[Tuple[TreeMemoryNode, int]]: Valid matched nodes and their effective match lengths.
        """
        max_depth = len(reverse_context)
        if max_depth == 0: 
            return[]
        
        # Dictionary acts as a deduplication cache, storing the path with the max effective length
        visited = {}  # Format: {id(node): (node, eff_len)}
        
        # --- 1. BASELINE SEARCH ('none') ---
        # Always executed. O(N) strict sequential backward search.
        curr_node = self.root
        for i in range(max_depth):
            token = reverse_context[i]
            if token not in curr_node.children: 
                break
            curr_node = curr_node.children[token]
            if i + 1 >= self.n_min:
                visited[id(curr_node)] = (curr_node, i + 1)
                
        # --- 2. ADVANCED MODES (Cumulative additions) ---
        if mode == 'linear':
            # Skip-Gram BFS. Phase 0 allows masks. Phase 1 requires strict matches.
            queue = deque([(self.root, 0, 0, 0)])  # (node, depth, phase, eff_len)
            while queue:
                curr_node, depth, phase, eff_len = queue.popleft()
                
                # Deduplicate: only overwrite if this masked path yields a longer effective match
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (curr_node, eff_len)
                        
                if depth == max_depth: 
                    continue
                    
                target_token = reverse_context[depth]
                
                if phase == 0:
                    for t, child in curr_node.children.items():
                        # Option A: Mask the current token (remain in Phase 0)
                        queue.append((child, depth + 1, 0, eff_len))
                        # Option B: Exact match locks the search into strict mode (Phase 1)
                        if t == target_token:
                            queue.append((child, depth + 1, 1, eff_len + 1))
                else: 
                    # Phase 1: Masking is disabled; unbroken chain required
                    if target_token in curr_node.children:
                        queue.append((curr_node.children[target_token], depth + 1, 1, eff_len + 1))
                        
        elif mode == 'squared':
            # Exhaustive Combinatorial BFS. Any token can be masked or matched.
            queue = deque([(self.root, 0, 0)])  # (node, depth, eff_len)
            while queue:
                curr_node, depth, eff_len = queue.popleft()
                
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (curr_node, eff_len)
                        
                if depth == max_depth: 
                    continue
                    
                target_token = reverse_context[depth]
                
                for t, child in curr_node.children.items():
                    # Increment effective length strictly upon positive hits
                    match_len = eff_len + 1 if t == target_token else eff_len
                    queue.append((child, depth + 1, match_len))
                    
        return list(visited.values())

    def predict_proba(self, 
                      temperature: Optional[float] = 1.0, 
                      top_k: Optional[int] = 0, 
                      top_p: Optional[float] = 1.0,
                      masked_mode: str = 'none') -> Dict[Any, float]:
        """
        Calculates the probability distribution for the next token based on context.

        Args:
            temperature (float): >1.0 increases randomness, <1.0 sharpens peaks.
            top_k (int): Keeps only the top K most likely tokens (0 to disable).
            top_p (float): Nucleus sampling threshold (1.0 to disable).
            masked_mode (str): Context matching strategy ('none', 'linear', 'squared').

        Returns:
            Dict[Any, float]: Normalized probability distribution of next possible tokens.
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
        
        # --- Probability Aggregation (Log-Space Context Mixing) ---
        for node, length in valid_nodes:
            delta = current_step - node.last_visit_step
            # Base node weight multiplier: Decay Penalty + Length Reward
            node_factor = (delta * log_decay_val) + (length * log_scale_base)
            
            for t, count in node.counts.items():
                if count <= 1e-9: 
                    continue
                found_pattern = True
                
                # Log(Count) + Factor
                log_weight = self._get_log_count(count) + node_factor
                curr = candidate_log_scores[t]
                
                if curr == -float('inf'):
                    candidate_log_scores[t] = log_weight
                else:
                    # Log-Sum-Exp trick prevents float underflow when merging branch probabilities
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
            val = math.exp(log_score - max_log)  # Shifted by max_log for stability
            linear_scores[token] = val
            total_sum += val
            
        probas = {t: v / total_sum for t, v in linear_scores.items()}
        
        # Early Exit Optimization if no filtering is requested
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
        """
        Samples a single token based on the internal probabilistic distribution.

        Args:
            temperature (float): >1.0 increases randomness, <1.0 sharpens peaks.
            top_k (int): Keeps only the top K most likely tokens.
            top_p (float): Nucleus sampling threshold.
            masked_mode (str): Context matching strategy.

        Returns:
            Optional[Any]: The predicted token, or None if the vocabulary is empty.
        """
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0
        
        # Extreme low temp flattens logic into greedy argmax
        if temp < 1e-4:
            probas = self.predict_proba(temperature=1.0, top_k=k, top_p=p, masked_mode=masked_mode)
            if not probas: 
                return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(temperature=temp, top_k=k, top_p=p, masked_mode=masked_mode)
        if not probas: 
            return None
            
        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    def update(self, actual: Any):
        """
        Ingests a new observation into the Suffix Trie model in O(N_max) operations.
        Applies Lazy Decay dynamically for currently active context branches.

        Args:
            actual (Any): The token observed in the stream.
        """
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
        
        hist_len = self.buffer.size
        history_tuple = self.buffer.to_tuple()
        node = self.root
        
        # Forward pass down the reversed sequence path (Suffix Trie building)
        for i in range(1, min(self.n_max, hist_len) + 1):
            token = history_tuple[-i]
            
            # Fast graph navigation & node construction
            node = node.children.setdefault(token, TreeMemoryNode())
            
            # Lazy Decay specific to the currently visited n-gram path
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
        Trains the model on a dataset. Automatically detects batch vs stream inputs.

        Args:
            X (Iterable): A stream of tokens or a batch of token sequences.
            verbose (bool): Whether to display a tqdm progress bar.

        Returns:
            TreeMemoryPredictor: self
        """
        is_batch = False
        
        # Safe heuristic to determine if X is a batch of sequences or a continuous stream
        if hasattr(X, '__len__') and len(X) > 0:
            first_element = next(iter(X))
            if isinstance(first_element, (list, tuple)) or (hasattr(first_element, '__iter__') and not isinstance(first_element, (str, bytes))):
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

    def update_context(self, token: Any): 
        """
        Manually pushes a token to the buffer without triggering weight updates.
        
        Args:
            token (Any): The token to append to context.
        """
        self.buffer.append(token)
        
    def fill_context(self, context: Iterable[Any]): 
        """
        Replaces the entire current context buffer.
        
        Args:
            context (Iterable[Any]): The new context sequence.
        """
        self.buffer.clear()
        self.buffer.extend(context)
        
    def reset_context(self): 
        """Flushes the current context buffer."""
        self.buffer.clear()
        
    # --- Serialization (Pickle Support) ---
    def __getstate__(self) -> Dict[str, Any]:
        """Purges dynamic caches strictly before serialization to save space."""
        state = self.__dict__.copy()
        for k in['_power_cache', '_int_log_cache']: 
            if k in state: 
                del state[k]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """Restores the object state and re-initializes all lazy caches appropriately."""
        self.__dict__.update(state)
        
        # Fallback bindings for older model variants
        if not hasattr(self, 'cache_size'): 
            self.cache_size = 4096
        if not hasattr(self, 'pruning_step'): 
            self.pruning_step = 1000
        
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
        """
        Saves the model instance to disk using pickle.

        Args:
            filepath (str): Destination file path.
        """
        try:
            with open(filepath, 'wb') as f: 
                pickle.dump(self, f)
        except Exception as e: 
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath: str) -> Optional['TreeMemoryPredictor']:
        """
        Loads a model instance from disk.

        Args:
            filepath (str): Source file path.

        Returns:
            Optional[TreeMemoryPredictor]: The loaded model, or None on failure.
        """
        try:
            with open(filepath, 'rb') as f: 
                return pickle.load(f)
        except Exception as e: 
            print(f"Error loading model: {e}")
            return None