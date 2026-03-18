import json
import math
import pickle
import random
from collections import defaultdict, deque
from typing import List, Dict, Optional, Iterable, Union, Tuple, Set, Any

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# Strict typing for the internal engine
Token = Union[str, int]


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

    def append(self, item: Token):
        """
        Appends an item, invalidates the tuple cache, and tracks size in O(1).
        
        Args:
            item (Token): The token to add to the buffer (str or int).
        """
        self._deque.append(item)
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Token]):
        """
        Extends the buffer with multiple items and synchronizes the size.
        
        Args:
            items (Iterable[Token]): Tokens to add.
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

    def to_tuple(self) -> Tuple[Token, ...]:
        """
        Returns an immutable tuple representation of the buffer.
        Caches the result to avoid redundant O(N) conversions.
        
        Returns:
            Tuple[Token, ...]: The current buffer state.
        """
        if self._cache_tuple is None:
            self._cache_tuple = tuple(self._deque)
        return self._cache_tuple

    # --- Persistence Methods ---
    def __getstate__(self) -> Dict[str, Union[int, deque]]:
        return {'_maxlen': self._maxlen, '_deque': self._deque, '_size': self._size}

    def __setstate__(self, state: Dict[str, Union[int, deque]]):
        self._maxlen = state.get('_maxlen', 10)
        self._deque = state.get('_deque', deque(maxlen=self._maxlen))
        self._size = state.get('_size', len(self._deque))
        self._cache_tuple = None


class TreeMemoryNode:
    """
    Lightweight Node for the Suffix Trie structure.
    Uses __slots__ to significantly reduce the memory footprint per instance.
    """
    __slots__ = ['counts', 'children', 'last_visit_step']
    
    def __init__(self):
        self.counts: Dict[Token, float] = defaultdict(float) 
        self.children: Dict[Token, 'TreeMemoryNode'] = {}
        self.last_visit_step: int = 0

    def to_dict(self) -> Dict[str, Union[Dict, int]]:
        """Serializes the node into a dictionary for safe JSON export."""
        return {
            'c': dict(self.counts),
            'ch': {str(k): v.to_dict() for k, v in self.children.items()},
            'lvs': self.last_visit_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[Dict, int]]) -> 'TreeMemoryNode':
        """Deserializes the node from a dictionary."""
        node = cls()
        node.counts = defaultdict(float, data.get('c', {}))
        node.children = {k: cls.from_dict(v) for k, v in data.get('ch', {}).items()}
        node.last_visit_step = data.get('lvs', 0)
        return node

    def __getstate__(self) -> Dict[str, Union[Dict, int]]:
        return {
            'counts': self.counts,
            'children': self.children,
            'last_visit_step': self.last_visit_step
        }

    def __setstate__(self, state: Dict[str, Union[Dict, int]]):
        self.counts = state.get('counts', defaultdict(float))
        self.children = state.get('children', {})
        self.last_visit_step = state.get('last_visit_step', 0)


class TreeMemoryPredictor:
    """
    Variable-order Markov Model utilizing a Reverse Suffix Trie.
    
    Features:
    - O(N) Traversal: Looks backward from the most recent token.
    - Lazy Decay: Weights decay mathematically only upon node visitation.
    - Katz-style Backoff: Interpolates unseen patterns with O(1) unigram fallbacks.
    - Log-Space Math: Prevents floating-point underflow on deep n-grams.
    - Skip-Grams / Masking: Supports wildcard sequence matching ('linear', 'squared').
    - Dynamic Garbage Collection: Adaptive pruning to bound memory usage.
    """

    def __init__(self, 
                 n_max: int = 10, 
                 n_min: int = 1, 
                 decay: float = 0.99, 
                 alphabet_autoscale: bool = True,
                 fallback_mode: str = 'katz_backoff',
                 pruning_mode: str = 'fixed',
                 pruning_step: int = 1000,
                 pruning_threshold: float = 1e-6,
                 max_beams: int = 1000,
                 cache_size: int = 4096):
        """
        Initializes the sequence predictor.

        Args:
            n_max (int): Maximum context length (n-gram order) to store and evaluate.
            n_min (int): Minimum effective context length required to accept a match.
            decay (float): Forgetting factor (0.0 to 1.0). Determines how fast old observations fade.
            alphabet_autoscale (bool): If True, scales weights by log(VocabSize) to balance entropy.
            fallback_mode (str): 'katz_backoff' (unigram smoothing) or 'uniform' (even probability).
            pruning_mode (str): 'fixed' (by step interval) or 'dynamic' (by tree size limits).
            pruning_step (int): Target interval or baseline threshold for the Garbage Collector.
            pruning_threshold (float): Weight threshold below which nodes are completely deleted.
            max_beams (int): Hard limit on nodes explored during 'linear' or 'squared' masked modes.
            cache_size (int): Maximum size for internal lazy math caches.
        """
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        self.fallback_mode = fallback_mode
        self.pruning_mode = pruning_mode
        self.pruning_step = pruning_step
        self.pruning_threshold = pruning_threshold
        self.max_beams = max_beams
        self.cache_size = cache_size
        
        # Internal counters and scaling variables
        self._vocab_len = 0 
        self._cached_log_base = 0.69314718056  # Initialized to log(2)
        self._last_computed_vocab_len = 0
        
        # Garbage Collection trackers
        self._node_count = 0
        self._next_prune_target = pruning_step

        # Katz Backoff Unigram trackers (O(1) continuous decay compliant)
        self.unigram_counts: Dict[Token, float] = defaultdict(float)
        self.unigram_last_update: Dict[Token, int] = defaultdict(int)

        # Lazy caches to avoid redundant heavy math operations
        self._power_cache: Dict[int, float] = {}
        self._power_cache_len = 0
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')

        self._int_log_cache: Dict[int, float] = {}
        self._log_cache_len = 0
        
        self.reset()

    def reset(self):
        """Resets the model to its initial empty state."""
        self.root = TreeMemoryNode()
        self.buffer = NBuffer(maxlen=self.n_max) 
        self.step = 0
        self.known_vocabulary: Set[Token] = set()
        self._vocab_len = 0
        self._node_count = 1
        self._next_prune_target = self.pruning_step
        
        self.unigram_counts.clear()
        self.unigram_last_update.clear()

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

    def _prune_recursive(self, node: TreeMemoryNode, current_step: int) -> int:
        """
        Garbage Collector step: Recursively applies true decay and removes empty branches.

        Args:
            node (TreeMemoryNode): The current Trie node.
            current_step (int): The global time step.

        Returns:
            int: The number of surviving nodes in this subtree.
        """
        delta = current_step - node.last_visit_step
        decay_factor = self._get_decay_factor(delta) if delta > 0 else 1.0
        
        keys_to_remove =[]
        for token, count in node.counts.items():
            real_count = count * decay_factor
            if real_count < self.pruning_threshold:
                keys_to_remove.append(token)
            else:
                node.counts[token] = real_count
                
        for token in keys_to_remove: 
            del node.counts[token]
            
        node.last_visit_step = current_step

        empty_children =[]
        surviving_nodes = 1  # Count self
        
        for token, child in node.children.items():
            child_survivors = self._prune_recursive(child, current_step)
            if not child.counts and not child.children:
                empty_children.append(token)
            else:
                surviving_nodes += child_survivors
                
        for token in empty_children: 
            del node.children[token]
            
        return surviving_nodes

    def prune_tree(self):
        """
        Triggers a Garbage Collection pass on the Suffix Trie and Unigram Backoff trackers.
        Automatically removes fully forgotten concepts from the vocabulary.
        """
        # 1. Prune the Suffix Trie
        surviving_nodes = self._prune_recursive(self.root, self.step)
        self._node_count = surviving_nodes
        
        # 2. Prune global unigram trackers to prevent OOV memory leaks
        keys_to_remove =[]
        for t, c in self.unigram_counts.items():
            delta = self.step - self.unigram_last_update.get(t, 0)
            val = c * (self._get_decay_factor(delta) if delta > 0 else 1.0)
            if val < self.pruning_threshold:
                keys_to_remove.append(t)
            else:
                self.unigram_counts[t] = val
                self.unigram_last_update[t] = self.step
                
        for t in keys_to_remove:
            del self.unigram_counts[t]
            del self.unigram_last_update[t]
            self.known_vocabulary.discard(t)
            
        self._vocab_len = len(self.known_vocabulary)

        # 3. Apply adaptive backoff scaling for Dynamic mode
        if self.pruning_mode == 'dynamic':
            self._next_prune_target = max(self.pruning_step, int(self._node_count * 1.5))

    def _get_context_nodes(self, mode: str, reverse_context: Tuple[Token, ...]) -> List[Tuple[TreeMemoryNode, int]]:
        """
        Searches the Reverse Suffix Trie based on the requested masking mode.

        Args:
            mode (str): Evaluation strategy ('none', 'linear', 'squared').
            reverse_context (Tuple[Token, ...]): Current buffer history reversed.

        Returns:
            List[Tuple[TreeMemoryNode, int]]: Valid matched nodes and their effective match lengths.
        """
        max_depth = len(reverse_context)
        if max_depth == 0: 
            return[]
        
        visited = {}  # Format: {id(node): (node, eff_len)}
        
        # --- 1. BASELINE SEARCH ('none') ---
        curr_node = self.root
        for i in range(max_depth):
            token = reverse_context[i]
            if token not in curr_node.children: 
                break
            curr_node = curr_node.children[token]
            if i + 1 >= self.n_min:
                visited[id(curr_node)] = (curr_node, i + 1)
                
        # --- 2. ADVANCED MODES (Cumulative additions with Beam limitations) ---
        if mode == 'linear':
            queue = deque([(self.root, 0, 0, 0)])  # (node, depth, phase, eff_len)
            beam_iters = 0
            
            while queue:
                beam_iters += 1
                if beam_iters > self.max_beams:
                    break
                    
                curr_node, depth, phase, eff_len = queue.popleft()
                
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (curr_node, eff_len)
                        
                if depth == max_depth: 
                    continue
                    
                target_token = reverse_context[depth]
                
                if phase == 0:
                    for t, child in curr_node.children.items():
                        queue.append((child, depth + 1, 0, eff_len))
                        if t == target_token:
                            queue.append((child, depth + 1, 1, eff_len + 1))
                else: 
                    if target_token in curr_node.children:
                        queue.append((curr_node.children[target_token], depth + 1, 1, eff_len + 1))
                        
        elif mode == 'squared':
            queue = deque([(self.root, 0, 0)])  # (node, depth, eff_len)
            beam_iters = 0
            
            while queue:
                beam_iters += 1
                if beam_iters > self.max_beams:
                    break
                    
                curr_node, depth, eff_len = queue.popleft()
                
                if depth > 0 and eff_len >= self.n_min:
                    nid = id(curr_node)
                    if eff_len > visited.get(nid, (None, -1))[1]: 
                        visited[nid] = (curr_node, eff_len)
                        
                if depth == max_depth: 
                    continue
                    
                target_token = reverse_context[depth]
                
                for t, child in curr_node.children.items():
                    match_len = eff_len + 1 if t == target_token else eff_len
                    queue.append((child, depth + 1, match_len))
                    
        return list(visited.values())

    def predict_proba(self, 
                      temperature: Optional[float] = 1.0, 
                      top_k: Optional[int] = 0, 
                      top_p: Optional[float] = 1.0,
                      masked_mode: str = 'none') -> Dict[Token, float]:
        """
        Calculates the probability distribution for the next token based on context.

        Args:
            temperature (float): >1.0 increases randomness, <1.0 sharpens peaks.
            top_k (int): Keeps only the top K most likely tokens (0 to disable).
            top_p (float): Nucleus sampling threshold (1.0 to disable).
            masked_mode (str): Context matching strategy ('none', 'linear', 'squared').

        Returns:
            Dict[Token, float]: Normalized probability distribution of next possible tokens.
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
        
        max_depth = min(self.n_max, hist_len)
        reverse_context = tuple(reversed(self.buffer.to_tuple()[-max_depth:]))
        
        valid_nodes = self._get_context_nodes(masked_mode, reverse_context)
        found_pattern = False
        
        # --- Probability Aggregation (Log-Space Context Mixing) ---
        for node, length in valid_nodes:
            delta = current_step - node.last_visit_step
            node_factor = (delta * log_decay_val) + (length * log_scale_base)
            
            for t, count in node.counts.items():
                if count <= 1e-9: 
                    continue
                found_pattern = True
                
                log_weight = self._get_log_count(count) + node_factor
                curr = candidate_log_scores[t]
                
                if curr == -float('inf'):
                    candidate_log_scores[t] = log_weight
                else:
                    if curr > log_weight: 
                        candidate_log_scores[t] = curr + math.log1p(math.exp(log_weight - curr))
                    else: 
                        candidate_log_scores[t] = log_weight + math.log1p(math.exp(curr - log_weight))

        # --- Base Fallback (Katz-style Backoff / Uniform) ---
        if not found_pattern:
            if self._vocab_len == 0: 
                return {}
            
            if self.fallback_mode == 'katz_backoff' and self.unigram_counts:
                # Interpolate using decayed global unigram frequencies (0-th order context)
                for t, c in self.unigram_counts.items():
                    delta = current_step - self.unigram_last_update.get(t, 0)
                    factor = self._get_decay_factor(delta) if delta > 0 else 1.0
                    val = c * factor
                    if val > 1e-9:
                        candidate_log_scores[t] = math.log(val)
            else:
                # Uniform fallback
                prob = 1.0 / self._vocab_len
                log_prob = math.log(prob)
                for tk in self.known_vocabulary:
                    candidate_log_scores[tk] = log_prob

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
        
        if k <= 0 and p >= 1.0:
            return dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        sorted_items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        
        if 0 < k < len(sorted_items): 
            sorted_items = sorted_items[:k]

        if p < 1.0:
            current_total_prob = sum(prob for _, prob in sorted_items)
            target_prob = p * current_total_prob 
            cumulative_prob = 0.0
            
            for i, (_, prob) in enumerate(sorted_items):
                cumulative_prob += prob
                if cumulative_prob >= target_prob:
                    sorted_items = sorted_items[:i + 1]
                    break

        new_total = sum(prob for _, prob in sorted_items)
        if new_total > 0: 
            return {tk: prob / new_total for tk, prob in sorted_items}
            
        return dict(sorted_items)

    def predict(self, 
                temperature: Optional[float] = 1.0, 
                top_k: Optional[int] = 0, 
                top_p: Optional[float] = 1.0,
                masked_mode: str = 'none') -> Optional[Token]:
        """
        Samples a single token based on the internal probabilistic distribution.

        Args:
            temperature (float): >1.0 increases randomness, <1.0 sharpens peaks.
            top_k (int): Keeps only the top K most likely tokens.
            top_p (float): Nucleus sampling threshold.
            masked_mode (str): Context matching strategy.

        Returns:
            Optional[Token]: The predicted token, or None if the vocabulary is empty.
        """
        temp = temperature if temperature is not None else 1.0
        k = top_k if top_k is not None else 0
        p = top_p if top_p is not None else 1.0
        
        if temp < 1e-4:
            probas = self.predict_proba(temperature=1.0, top_k=k, top_p=p, masked_mode=masked_mode)
            if not probas: 
                return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(temperature=temp, top_k=k, top_p=p, masked_mode=masked_mode)
        if not probas: 
            return None
            
        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    def _validate_token(self, token: Any):
        """Strictly validates that the token is an int or str, explicitly rejecting booleans."""
        if not isinstance(token, (str, int)) or isinstance(token, bool):
            raise TypeError(f"TreeMemoryPredictor strictly accepts 'str' or 'int' tokens. Got: {type(token).__name__} ({token})")

    def update(self, actual: Token):
        """
        Ingests a new observation into the Suffix Trie model in O(N_max) operations.
        Applies Lazy Decay dynamically for currently active context branches.

        Args:
            actual (Token): The token observed in the stream (str or int).
        """
        self._validate_token(actual)
        
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
            
        # O(1) Unigram Tracking for Katz-style Backoff Smoothing
        delta_uni = current_step - self.unigram_last_update.get(actual, 0)
        if delta_uni > 0 and actual in self.unigram_counts:
            self.unigram_counts[actual] *= self._get_decay_factor(delta_uni)
        self.unigram_counts[actual] += 1.0
        self.unigram_last_update[actual] = current_step
        
        hist_len = self.buffer.size
        history_tuple = self.buffer.to_tuple()
        node = self.root
        
        for i in range(1, min(self.n_max, hist_len) + 1):
            token = history_tuple[-i]
            
            if token not in node.children:
                node.children[token] = TreeMemoryNode()
                self._node_count += 1
            node = node.children[token]
            
            if node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = self._get_decay_factor(delta)
                    keys_to_remove =[]
                    
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < self.pruning_threshold: 
                            keys_to_remove.append(t)
                        else: 
                            node.counts[t] = new_val
                            
                    for t in keys_to_remove: 
                        del node.counts[t]
            
            node.last_visit_step = current_step
            node.counts[actual] += 1.0
            
        self.buffer.append(actual)
        
        if self.pruning_mode == 'fixed':
            if self.step % self.pruning_step == 0: 
                self.prune_tree()
        elif self.pruning_mode == 'dynamic':
            if self._node_count >= self._next_prune_target:
                self.prune_tree()

    def fit(self, X: Union[Iterable[Token], Iterable[Iterable[Token]]], verbose: bool = True):
        """
        Trains the model on a dataset. Automatically detects batch vs stream inputs.

        Args:
            X (Iterable): A stream of tokens or a batch of token sequences.
            verbose (bool): Whether to display a tqdm progress bar.

        Returns:
            TreeMemoryPredictor: self
        """
        is_batch = False
        
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

    def update_context(self, token: Token): 
        """Pushes a token to the buffer without triggering weight updates."""
        self._validate_token(token)
        self.buffer.append(token)
        
    def fill_context(self, context: Iterable[Token]): 
        """Replaces the entire current context buffer."""
        for token in context:
            self._validate_token(token)
        self.buffer.clear()
        self.buffer.extend(context)
        
    def reset_context(self): 
        """Flushes the current context buffer."""
        self.buffer.clear()

    # --- Safe Serialization (JSON) ---
    def to_dict(self) -> Dict[str, Any]:
        """
        Exports the entire model state to a JSON-serializable dictionary.
        Requires tokens to be castable to strings.
        """
        return {
            'n_max': self.n_max,
            'n_min': self.n_min,
            'decay': self.decay,
            'alphabet_autoscale': self.alphabet_autoscale,
            'fallback_mode': self.fallback_mode,
            'pruning_mode': self.pruning_mode,
            'pruning_step': self.pruning_step,
            'pruning_threshold': self.pruning_threshold,
            'max_beams': self.max_beams,
            'step': self.step,
            'known_vocabulary': list(self.known_vocabulary),
            'unigram_counts': {str(k): v for k, v in self.unigram_counts.items()},
            'unigram_last_update': {str(k): v for k, v in self.unigram_last_update.items()},
            'buffer': list(self.buffer._deque),
            'root': self.root.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeMemoryPredictor':
        """Restores the model from a dictionary payload."""
        model = cls(
            n_max=data.get('n_max', 10),
            n_min=data.get('n_min', 1),
            decay=data.get('decay', 0.99),
            alphabet_autoscale=data.get('alphabet_autoscale', True),
            fallback_mode=data.get('fallback_mode', 'katz_backoff'),
            pruning_mode=data.get('pruning_mode', 'fixed'),
            pruning_step=data.get('pruning_step', 1000),
            pruning_threshold=data.get('pruning_threshold', 1e-6),
            max_beams=data.get('max_beams', 1000)
        )
        model.step = data.get('step', 0)
        model.known_vocabulary = set(data.get('known_vocabulary',[]))
        model._vocab_len = len(model.known_vocabulary)
        model.unigram_counts = defaultdict(float, data.get('unigram_counts', {}))
        model.unigram_last_update = defaultdict(int, data.get('unigram_last_update', {}))
        model.buffer.extend(data.get('buffer',[]))
        model.root = TreeMemoryNode.from_dict(data.get('root', {}))
        return model

    def save_json(self, filepath: str):
        """Saves the model state safely to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, filepath: str) -> 'TreeMemoryPredictor':
        """Loads a model safely from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # --- Legacy Serialization (Pickle Support) ---
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for k in['_power_cache', '_int_log_cache']: 
            if k in state: 
                del state[k]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        
        if getattr(self, 'pruning_mode', None) is None: 
            self.pruning_mode = 'fixed'
        if not hasattr(self, 'max_beams'): 
            self.max_beams = 1000
        if not hasattr(self, 'pruning_threshold'):
            self.pruning_threshold = 1e-6
        if not hasattr(self, 'fallback_mode'):
            self.fallback_mode = 'katz_backoff'
        if not hasattr(self, '_node_count'):
            self._node_count = 1
            self._next_prune_target = getattr(self, 'pruning_step', 1000)
            
        if not hasattr(self, 'unigram_counts'):
            self.unigram_counts = defaultdict(float)
            self.unigram_last_update = defaultdict(int)
            for tk in getattr(self, 'known_vocabulary', set()):
                self.unigram_counts[tk] = 1.0
                self.unigram_last_update[tk] = getattr(self, 'step', 0)
        
        self._power_cache = {}
        self._power_cache_len = 0
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')
        self._int_log_cache = {} 
        self._log_cache_len = 0
        self._last_computed_vocab_len = 0

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f: 
                pickle.dump(self, f)
        except Exception as e: 
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath: str) -> Optional['TreeMemoryPredictor']:
        try:
            with open(filepath, 'rb') as f: 
                return pickle.load(f)
        except Exception as e: 
            print(f"Error loading model: {e}")
            return None