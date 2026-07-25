import json
import math
import pickle
import random
import warnings
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any, Iterable, Union, Tuple, Set

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# Strict typing for the internal engine (Strings or Integers only)
Token = Union[str, int]


class TokenBuffer:
    """Optimized stateful buffer for sliding window operations.

    Wraps `collections.deque` and maintains a manual size counter to bypass 
    O(N) operations when continuously checking size or converting to tuples 
    in hot loops.

    Attributes:
        _maxlen (int): The maximum number of items the buffer can hold. Must be >= 1.
        _deque (deque): Double-ended queue storing the token history.
        _cache_tuple (Optional[Tuple[Token, ...]]): Cached tuple representation of the buffer.
        _size (int): Current number of elements inside the buffer. Range: [0, _maxlen].
    """
    __slots__ = ['_maxlen', '_deque', '_cache_tuple', '_size']

    def __init__(self, maxlen: int):
        """Initializes the token buffer.

        Args:
            maxlen (int): The maximum number of items the buffer can hold. Range: [1, inf).
        """
        self._maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._cache_tuple = None
        self._size = 0  

    def append(self, item: Token):
        """Appends a token, invalidating the tuple cache and tracking size in O(1).
        
        Args:
            item (Token): The token to add to the buffer (str or int).
        """
        self._deque.append(item)
        self._cache_tuple = None
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: Iterable[Token]):
        """Extends the buffer with multiple tokens and updates the size.
        
        Args:
            items (Iterable[Token]): Tokens to add (iterable of str or int).
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
        """Returns the current size of the buffer.
        
        Returns:
            int: Number of elements in the buffer (O(1) operation). Range: [0, maxlen].
        """
        return self._size

    def to_tuple(self) -> Tuple[Token, ...]:
        """Returns an immutable tuple representation of the buffer.
        
        Returns:
            Tuple[Token, ...]: The current cached buffer state.
        """
        if self._cache_tuple is None:
            self._cache_tuple = tuple(self._deque)
        return self._cache_tuple

    def __getstate__(self) -> Dict[str, Union[int, deque]]:
        """Extracts the state dictionary for serialization."""
        return {'_maxlen': self._maxlen, '_deque': self._deque, '_size': self._size}

    def __setstate__(self, state: Dict[str, Union[int, deque]]):
        """Restores state variables after deserialization."""
        self._maxlen = state.get('_maxlen', 10)
        self._deque = state.get('_deque', deque(maxlen=self._maxlen))
        self._size = state.get('_size', len(self._deque))
        self._cache_tuple = None


class TreeMemoryNode:
    """Lightweight Node for the Suffix Trie structure.

    Attributes:
        counts (Dict[Token, float]): Map from a predicted next token to its frequency.
            Frequencies must be >= 0.0.
        children (Dict[Token, TreeMemoryNode]): Children subtrees representing preceding context tokens.
        last_visit_step (int): Global timeline step of the last update to this node. Range: [0, inf).
    """
    __slots__ = ['counts', 'children', 'last_visit_step']
    
    def __init__(self):
        """Initializes an empty Trie node."""
        self.counts: Dict[Token, float] = defaultdict(float) 
        self.children: Dict[Token, 'TreeMemoryNode'] = {}
        self.last_visit_step: int = 0

    def to_dict(self) -> Dict[str, Union[Dict, int]]:
        """Serializes the node into a JSON-safe dictionary.

        Returns:
            Dict[str, Union[Dict, int]]: Serialized node data.
        """
        return {
            'c': dict(self.counts),
            'ch': {str(k): v.to_dict() for k, v in self.children.items()},
            'lvs': self.last_visit_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[Dict, int]]) -> 'TreeMemoryNode':
        """Deserializes a node from a state dictionary.

        Args:
            data (Dict[str, Union[Dict, int]]): Dictionary of serialized node parameters.

        Returns:
            TreeMemoryNode: Deserialized node instance.
        """
        node = cls()
        node.counts = defaultdict(float, data.get('c', {}))
        node.children = {k: cls.from_dict(v) for k, v in data.get('ch', {}).items()}
        node.last_visit_step = data.get('lvs', 0)
        return node

    def __getstate__(self) -> Dict[str, Union[Dict, int]]:
        """Extracts state parameters for pickle serialization."""
        return {
            'counts': self.counts,
            'children': self.children,
            'last_visit_step': self.last_visit_step
        }

    def __setstate__(self, state: Dict[str, Union[Dict, int]]):
        """Restores state parameters from a pickle payload."""
        self.counts = state.get('counts', defaultdict(float))
        self.children = state.get('children', {})
        self.last_visit_step = state.get('last_visit_step', 0)


class TreeMemoryPredictor:
    """Variable-order Markov Model utilizing a Reverse Suffix Trie.

    Provides O(N) context traversal looking backwards, lazy weight decay 
    applied strictly on node visitation, Katz-style backoff fallback, 
    dynamic memory garbage collection, and asymmetric federated merging.

    Attributes:
        n_max (int): Maximum depth of context (Markov order). Range: [1, inf).
        n_min (int): Minimum context length required for a valid match. Range: [1, inf).
        decay (float): Exponential forgetting rate. Range: [0.0, 1.0].
        alphabet_autoscale (bool): Whether to scale weights dynamically based on vocabulary size.
        fallback_mode (str): Smoothing technique when no context matches. Values in {'katz_backoff', 'uniform'}.
        pruning_mode (str): Strategy for tree garbage collection. Values in {'fixed', 'dynamic'}.
        pruning_step (int): Target interval or threshold for garbage collection. Range: [1, inf).
        pruning_threshold (float): Weight threshold below which nodes/counts are pruned. Range: [0.0, inf).
        max_beams (int): Upper bound on explored paths during masked search modes. Range: [1, inf).
        cache_size (int): Maximum capacity of the lazy math caches. Range: [1, inf).
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
        """Initializes the sequence predictor with hyperparameter constraints.

        Args:
            n_max (int): Maximum depth of context. Range: [1, inf). Defaults to 10.
            n_min (int): Minimum context length required. Range: [1, inf). Defaults to 1.
            decay (float): Exponential forgetting rate. Range: [0.0, 1.0]. Defaults to 0.99.
            alphabet_autoscale (bool): Enables dynamic entropy scaling. Defaults to True.
            fallback_mode (str): Fallback smoothing mode. Values in {'katz_backoff', 'uniform'}. Defaults to 'katz_backoff'.
            pruning_mode (str): GC strategy. Values in {'fixed', 'dynamic'}. Defaults to 'fixed'.
            pruning_step (int): Pruning step target. Range: [1, inf). Defaults to 1000.
            pruning_threshold (float): Limit below which weights are deleted. Range: [0.0, inf). Defaults to 1e-6.
            max_beams (int): Bounded path limit in masked search. Range: [1, inf). Defaults to 1000.
            cache_size (int): Math cache capacity limit. Range: [1, inf). Defaults to 4096.
        """
        # --- Parameter Validation & Fallback Logic ---
        if not isinstance(n_max, int) or n_max < 1:
            warnings.warn(f"Invalid n_max={n_max}. Must be an integer >= 1. Falling back to default: 10.")
            n_max = 10
            
        if not isinstance(n_min, int) or n_min < 1:
            warnings.warn(f"Invalid n_min={n_min}. Must be an integer >= 1. Falling back to default: 1.")
            n_min = 1
            
        if not isinstance(decay, (int, float)) or not (0.0 <= decay <= 1.0):
            warnings.warn(f"Invalid decay={decay}. Must be a float in range [0.0, 1.0]. Falling back to default: 0.99.")
            decay = 0.99
            
        valid_fallbacks = {'katz_backoff', 'uniform'}
        if fallback_mode not in valid_fallbacks:
            warnings.warn(f"Invalid fallback_mode='{fallback_mode}'. Choose from {sorted(list(valid_fallbacks))}. Falling back to default: 'katz_backoff'.")
            fallback_mode = 'katz_backoff'
            
        valid_pruning = {'fixed', 'dynamic'}
        if pruning_mode not in valid_pruning:
            warnings.warn(f"Invalid pruning_mode='{pruning_mode}'. Choose from {sorted(list(valid_pruning))}. Falling back to default: 'fixed'.")
            pruning_mode = 'fixed'
            
        if not isinstance(pruning_step, int) or pruning_step < 1:
            warnings.warn(f"Invalid pruning_step={pruning_step}. Must be an integer >= 1. Falling back to default: 1000.")
            pruning_step = 1000
            
        if not isinstance(pruning_threshold, (int, float)) or pruning_threshold < 0.0:
            warnings.warn(f"Invalid pruning_threshold={pruning_threshold}. Must be a float >= 0.0. Falling back to default: 1e-6.")
            pruning_threshold = 1e-6
            
        if not isinstance(max_beams, int) or max_beams < 1:
            warnings.warn(f"Invalid max_beams={max_beams}. Must be an integer >= 1. Falling back to default: 1000.")
            max_beams = 1000
            
        if not isinstance(cache_size, int) or cache_size < 1:
            warnings.warn(f"Invalid cache_size={cache_size}. Must be an integer >= 1. Falling back to default: 4096.")
            cache_size = 4096

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
        
        self._vocab_len = 0 
        self._cached_log_base = 0.69314718056  # Precalculated ln(2)
        self._last_computed_vocab_len = 0
        
        self._node_count = 0
        self._next_prune_target = pruning_step

        self.unigram_counts: Dict[Token, float] = defaultdict(float)
        self.unigram_last_update: Dict[Token, int] = defaultdict(int)

        self._power_cache: Dict[int, float] = {}
        self._power_cache_len = 0
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')

        self._int_log_cache: Dict[int, float] = {}
        self._log_cache_len = 0
        
        self.reset()

    def reset(self):
        """Resets the model back to an empty initial state.

        Returns:
            TreeMemoryPredictor: Self instance for method chaining.
        """
        self.root = TreeMemoryNode()
        self.buffer = TokenBuffer(maxlen=self.n_max) 
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
        """Computes or retrieves the dynamic scaling factor based on vocabulary size.

        Balances long context matching weight against the entropy of the alphabet.

        Returns:
            float: Logarithmic scaling base (minimum ln(2)).
        """
        if not self.alphabet_autoscale:
            return 0.69314718056
        
        if self._vocab_len != self._last_computed_vocab_len:
             self._last_computed_vocab_len = self._vocab_len
             self._cached_log_base = math.log(max(2, self._vocab_len))
             
        return self._cached_log_base

    def _get_decay_factor(self, delta: int) -> float:
        """Lazily calculates and caches exponential decay multipliers.

        Args:
            delta (int): Elapsed time steps since last update. Range: [0, inf).

        Returns:
            float: The computed decay factor (decay ^ delta). Range: [0.0, 1.0].
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
        """Computes and caches the natural logarithm optimized for integer-like counts.

        Args:
            count (float): Token frequency count. Range: [0.0, inf).

        Returns:
            float: Natural logarithm of the count. Range: [0.0, inf).
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
        """Recursively updates decay states and removes empty/sub-threshold branches.

        Args:
            node (TreeMemoryNode): Current subtree node.
            current_step (int): Global timeline step of the model. Range: [1, inf).

        Returns:
            int: Number of surviving nodes within this subtree. Range: [1, inf).
        """
        delta = current_step - node.last_visit_step
        decay_factor = self._get_decay_factor(delta) if delta > 0 else 1.0
        
        # 1. Decay and drop sub-threshold token frequencies
        keys_to_remove = []
        for token, count in node.counts.items():
            real_count = count * decay_factor
            if real_count < self.pruning_threshold:
                keys_to_remove.append(token)
            else:
                node.counts[token] = real_count
                
        for token in keys_to_remove: 
            del node.counts[token]
            
        node.last_visit_step = current_step

        # 2. Process subtrees and prune dead structures
        empty_children = []
        surviving_nodes = 1
        
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
        """Triggers a complete Garbage Collection sweep over the Trie and unigrams.

        Permanently deletes paths and vocabulary items with weights falling
        below the configured pruning threshold.
        """
        surviving_nodes = self._prune_recursive(self.root, self.step)
        self._node_count = surviving_nodes
        
        keys_to_remove = []
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

        if self.pruning_mode == 'dynamic':
            self._next_prune_target = max(self.pruning_step, int(self._node_count * 1.5))

    def _get_context_nodes(self, mode: str, reverse_context: Tuple[Token, ...]) -> List[Tuple[TreeMemoryNode, int]]:
        """Queries the Reverse Suffix Trie using the designated evaluation strategy.

        Args:
            mode (str): Searching strategy. Values in {'none', 'linear', 'squared'}.
            reverse_context (Tuple[Token, ...]): Chronologically inverted context history.

        Returns:
            List[Tuple[TreeMemoryNode, int]]: List of matched nodes paired with their effective lengths.
        """
        max_depth = len(reverse_context)
        if max_depth == 0: 
            return []
        
        visited = {}  # Format: {id(node): (node, eff_len)}
        
        # --- 1. Exact Match Search ('none') ---
        curr_node = self.root
        for i in range(max_depth):
            token = reverse_context[i]
            if token not in curr_node.children: 
                break
            curr_node = curr_node.children[token]
            if i + 1 >= self.n_min:
                visited[id(curr_node)] = (curr_node, i + 1)
                
        # --- 2. Bounded Skip-Recent-Noise Search ('linear') ---
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
                        
        # --- 3. Full Combinatorial Match Search ('squared') ---
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

    def _validate_inference_params(self, 
                                   temperature: Union[float, str, None], 
                                   top_k: Union[int, str, None], 
                                   top_p: Union[float, str, None], 
                                   masked_mode: str) -> Tuple[float, int, float, str]:
        """Validates and coerces inference parameters with fallback warnings.

        Args:
            temperature (Union[float, str, None]): Generation temperature.
                Accepts a float in the range [0.0, inf), or "none"/None to disable 
                scaling (equivalent to 0.0, enabling deterministic greedy argmax decoding).
            top_k (Union[int, str, None]): Top-K filter constraint.
                Accepts an integer in the range [0, inf), or "none"/None to disable 
                filtering (equivalent to 0).
            top_p (Union[float, str, None]): Nucleus sampling constraint.
                Accepts a float in the range [0.0, 1.0], or "none"/None to disable 
                filtering (equivalent to 1.0).
            masked_mode (str): Matching strategy mode. Values in {'none', 'linear', 'squared'}.

        Returns:
            Tuple[float, int, float, str]: Validated and coerced parameters.
        """
        # --- Convert "none" / None strings into neutral mathematical fallbacks ---
        temp = 0.0 if temperature in (None, "none", "None") else temperature
        k = 0 if top_k in (None, "none", "None") else top_k
        p = 1.0 if top_p in (None, "none", "None") else top_p

        if not isinstance(temp, (int, float)) or temp < 0.0:
            warnings.warn(f"Invalid temperature={temperature}. Must be a float >= 0.0 or 'none'. Falling back to 0.0.")
            temp = 0.0

        if not isinstance(k, int) or k < 0:
            warnings.warn(f"Invalid top_k={top_k}. Must be an integer >= 0 or 'none'. Falling back to 0 (disabled).")
            k = 0

        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            warnings.warn(f"Invalid top_p={top_p}. Must be a float in range [0.0, 1.0] or 'none'. Falling back to 1.0.")
            p = 1.0

        valid_modes = {'none', 'linear', 'squared'}
        if masked_mode not in valid_modes:
            warnings.warn(f"Invalid masked_mode='{masked_mode}'. Choose from {sorted(list(valid_modes))}. Falling back to default: 'none'.")
            masked_mode = 'none'

        return temp, k, p, masked_mode

    def predict_proba(self, 
                      temperature: Union[float, str, None] = "none", 
                      top_k: Union[int, str, None] = "none", 
                      top_p: Union[float, str, None] = "none",
                      masked_mode: str = 'none',
                      *,
                      _validated: bool = False) -> Dict[Token, float]:
        """Calculates the probability distribution for the next token based on context.

        Args:
            temperature (Union[float, str, None]): Adjusts distribution flatness. 
                Accepts a float in the range [0.0, inf), or "none"/None to disable 
                scaling (equivalent to 0.0, enabling deterministic greedy argmax decoding).
            top_k (Union[int, str, None]): Keeps only the top K highest probability candidates.
                Accepts an integer in the range [0, inf), or "none"/None to disable 
                filtering (equivalent to 0).
            top_p (Union[float, str, None]): Nucleus sampling threshold to retain top cumulative mass.
                Accepts a float in the range [0.0, 1.0], or "none"/None to disable 
                filtering (equivalent to 1.0).
            masked_mode (str): Evaluation strategy. Values in {'none', 'linear', 'squared'}.
            _validated (bool): Internal bypass flag to avoid redundant warning triggers.

        Returns:
            Dict[Token, float]: Normalised probability distribution sorted in descending order.
        """
        if not _validated:
            temperature, top_k, top_p, masked_mode = self._validate_inference_params(
                temperature, top_k, top_p, masked_mode
            )

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
        
        # Accumulate context scores inside the log space
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

        if not found_pattern:
            if self._vocab_len == 0: 
                return {}
            
            if self.fallback_mode == 'katz_backoff' and self.unigram_counts:
                for t, c in self.unigram_counts.items():
                    delta = current_step - self.unigram_last_update.get(t, 0)
                    factor = self._get_decay_factor(delta) if delta > 0 else 1.0
                    val = c * factor
                    if val > 1e-9:
                        candidate_log_scores[t] = math.log(val)
            else:
                prob = 1.0 / self._vocab_len
                log_prob = math.log(prob)
                for tk in self.known_vocabulary:
                    candidate_log_scores[tk] = log_prob

        if temperature != 1.0 and temperature > 1e-4:
            for t in candidate_log_scores: 
                candidate_log_scores[t] /= temperature

        max_log = max(candidate_log_scores.values())
        linear_scores = {}
        total_sum = 0.0
        
        for token, log_score in candidate_log_scores.items():
            val = math.exp(log_score - max_log)
            linear_scores[token] = val
            total_sum += val
            
        probas = {t: v / total_sum for t, v in linear_scores.items()}
        
        if top_k <= 0 and top_p >= 1.0:
            return dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        sorted_items = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        
        if 0 < top_k < len(sorted_items): 
            sorted_items = sorted_items[:top_k]

        if top_p < 1.0:
            current_total_prob = sum(prob for _, prob in sorted_items)
            target_prob = top_p * current_total_prob 
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
                temperature: Union[float, str, None] = "none", 
                top_k: Union[int, str, None] = "none", 
                top_p: Union[float, str, None] = "none",
                masked_mode: str = 'none') -> Optional[Token]:
        """Samples a single token based on the internal probabilistic distribution.

        Args:
            temperature (Union[float, str, None]): Controls prediction randomness.
                Accepts a float in the range [0.0, inf), or "none"/None to disable 
                scaling (equivalent to 0.0, enabling deterministic greedy argmax decoding).
            top_k (Union[int, str, None]): Restricts sampling to top K candidates.
                Accepts an integer in the range [0, inf), or "none"/None to disable 
                filtering (equivalent to 0).
            top_p (Union[float, str, None]): Restricts sampling to nucleus probability mass.
                Accepts a float in the range [0.0, 1.0], or "none"/None to disable 
                filtering (equivalent to 1.0).
            masked_mode (str): Path search strategy. Values in {'none', 'linear', 'squared'}.

        Returns:
            Optional[Token]: The predicted next token, or None if the vocabulary is empty.
        """
        temperature, top_k, top_p, masked_mode = self._validate_inference_params(
            temperature, top_k, top_p, masked_mode
        )
        
        if temperature < 1e-4:
            probas = self.predict_proba(
                temperature=1.0, top_k=top_k, top_p=top_p, masked_mode=masked_mode, _validated=True
            )
            if not probas: 
                return None
            return max(probas, key=probas.get)
        
        probas = self.predict_proba(
            temperature=temperature, top_k=top_k, top_p=top_p, masked_mode=masked_mode, _validated=True
        )
        if not probas: 
            return None
            
        return random.choices(list(probas.keys()), weights=list(probas.values()), k=1)[0]

    def _validate_token(self, token: Any):
        """Ensures the token type is either string or integer (excluding boolean).

        Args:
            token (Any): Target token to validate.

        Raises:
            TypeError: If the token type is invalid.
        """
        if not isinstance(token, (str, int)) or isinstance(token, bool):
            raise TypeError(f"TreeMemoryPredictor strictly accepts 'str' or 'int' tokens. Got: {type(token).__name__} ({token})")

    def update(self, actual: Token):
        """Ingests a new token into the sequence stream and updates Trie statistics.

        Applies lazy decay strictly to the active context branches in O(N_max) operations.

        Args:
            actual (Token): The newly observed token (str or int).
        """
        self._validate_token(actual)
        
        self.step += 1
        current_step = self.step
        
        if actual not in self.known_vocabulary:
            self.known_vocabulary.add(actual)
            self._vocab_len += 1
            
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
                    keys_to_remove = []
                    
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
        """Trains the model on a flat stream of tokens or on independent batch sequences.

        Args:
            X (Union[Iterable[Token], Iterable[Iterable[Token]]]): Token stream or sequence batch.
            verbose (bool): Whether to display a progress bar.

        Returns:
            TreeMemoryPredictor: Self instance.
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

    def _merge_recursive(self, node_self: TreeMemoryNode, node_other: TreeMemoryNode, current_step_self: int, current_step_other: int, other_model: 'TreeMemoryPredictor'):
        """Recursively projects external Trie weights into the local timeline.

        Args:
            node_self (TreeMemoryNode): Target node of the local model.
            node_other (TreeMemoryNode): Source node of the external model.
            current_step_self (int): Current global timeline step of self.
            current_step_other (int): Current global timeline step of other.
            other_model (TreeMemoryPredictor): Reference to the foreign model.
        """
        delta_self = current_step_self - node_self.last_visit_step
        factor_self = self._get_decay_factor(delta_self) if delta_self > 0 else 1.0
        
        delta_other = current_step_other - node_other.last_visit_step
        factor_other = other_model._get_decay_factor(delta_other) if delta_other > 0 else 1.0
        
        for t in list(node_self.counts.keys()): 
            node_self.counts[t] *= factor_self
            
        for t, c in node_other.counts.items():
            node_self.counts[t] += (c * factor_other)
            
        node_self.last_visit_step = current_step_self

        for t, child_other in node_other.children.items():
            if t not in node_self.children:
                node_self.children[t] = TreeMemoryNode()
            self._merge_recursive(node_self.children[t], child_other, current_step_self, current_step_other, other_model)

    def merge(self, other: 'TreeMemoryPredictor') -> 'TreeMemoryPredictor':
        """Merges another model's learned state into this one (Federated Learning).

        Adapts depth and context bounds, resolving timeline step shifts 
        using mathematical projections of decayed states.

        Args:
            other (TreeMemoryPredictor): External model instance to absorb.

        Returns:
            TreeMemoryPredictor: Updated self instance.
        """
        if other.n_max > self.n_max:
            self.n_max = other.n_max
            new_buffer = TokenBuffer(maxlen=self.n_max)
            new_buffer.extend(self.buffer.to_tuple())
            self.buffer = new_buffer
            
        if other.n_min < self.n_min:
            self.n_min = other.n_min
            
        self.known_vocabulary.update(other.known_vocabulary)
        
        for t, c in other.unigram_counts.items():
            delta_other = other.step - other.unigram_last_update.get(t, 0)
            true_weight_other = c * (other._get_decay_factor(delta_other) if delta_other > 0 else 1.0)
            
            delta_self = self.step - self.unigram_last_update.get(t, 0)
            true_weight_self = self.unigram_counts.get(t, 0.0) * (self._get_decay_factor(delta_self) if delta_self > 0 else 1.0)
            
            self.unigram_counts[t] = true_weight_self + true_weight_other
            self.unigram_last_update[t] = self.step

        self._merge_recursive(self.root, other.root, self.step, other.step, other)
        
        self.prune_tree()
        return self

    def update_context(self, token: Token): 
        """Pushes a token to the buffer without triggering weight updates.

        Args:
            token (Token): Context token to append.
        """
        self._validate_token(token)
        self.buffer.append(token)
        
    def fill_context(self, context: Iterable[Token]): 
        """Replaces the entire sliding window context with a new sequence.

        Args:
            context (Iterable[Token]): Source sequence of tokens.
        """
        for token in context:
            self._validate_token(token)
        self.buffer.clear()
        self.buffer.extend(context)
        
    def reset_context(self): 
        """Flushes the current context buffer."""
        self.buffer.clear()

    # --- Safe Serialization (JSON) ---
    def to_dict(self) -> Dict[str, Any]:
        """Exports the entire model state to a JSON-compatible dictionary.

        Returns:
            Dict[str, Any]: Serialized dictionary of the model state.
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
        """Restores the model from a dictionary state representation.

        Args:
            data (Dict[str, Any]): Dictionary of serialized state.

        Returns:
            TreeMemoryPredictor: Deserialized model instance.
        """
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
        model.known_vocabulary = set(data.get('known_vocabulary', []))
        model._vocab_len = len(model.known_vocabulary)
        model.unigram_counts = defaultdict(float, data.get('unigram_counts', {}))
        model.unigram_last_update = defaultdict(int, data.get('unigram_last_update', {}))
        model.buffer.extend(data.get('buffer', []))
        model.root = TreeMemoryNode.from_dict(data.get('root', {}))
        return model

    def save_json(self, filepath: str):
        """Saves the model state safely to a JSON file.

        Args:
            filepath (str): Target output file path.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, filepath: str) -> 'TreeMemoryPredictor':
        """Restores a model state from a JSON file.

        Args:
            filepath (str): Input file path.

        Returns:
            TreeMemoryPredictor: Deserialized model instance.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # --- Legacy Serialization (Pickle Support) ---
    def __getstate__(self) -> Dict[str, Any]:
        """Strips volatile mathematical runtime caches before pickle serialization.

        Returns:
            Dict[str, Any]: Copy of the state dictionary.
        """
        state = self.__dict__.copy()
        for k in ['_power_cache', '_int_log_cache']: 
            if k in state: 
                del state[k]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """Restores model state from pickle payload and re-initializes volatile caches.

        Args:
            state (Dict[str, Any]): Dict of pickle state.
        """
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
        """Saves the model instance to a binary pickle file.

        Args:
            filepath (str): Output file path.
        """
        try:
            with open(filepath, 'wb') as f: 
                pickle.dump(self, f)
        except Exception as e: 
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath: str) -> Optional['TreeMemoryPredictor']:
        """Loads a model instance from a binary pickle file.

        Args:
            filepath (str): Input file path.

        Returns:
            Optional[TreeMemoryPredictor]: Loaded model instance, or None if an error occurs.
        """
        try:
            with open(filepath, 'rb') as f: 
                return pickle.load(f)
        except Exception as e: 
            print(f"Error loading model: {e}")
            return None