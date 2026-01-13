import math
import pickle
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any, Iterable, Union

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

class TreeMemoryNode:
    """Trie Node using __slots__ for minimal memory footprint."""
    __slots__ = ['counts', 'children', 'last_visit_step']
    def __init__(self):
        self.counts = defaultdict(float) # Linear counts for fast updates
        self.children = {}
        self.last_visit_step = 0

class TreeMemoryPredictor:
    """
    Log-Space Context Mixing Model (TMP).
    Features: Entropy Scaling (S^L), Lazy Decay, Batch Processing.
    """
    def __init__(self, n_max: int = 10, n_min: int = 1, decay: float = 0.99, alphabet_autoscale: bool = True):
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        
        # Pre-calculate log decay to avoid math.log() calls in loops
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')
        self.reset()

    def reset(self):
        self.root = TreeMemoryNode()
        self.history = deque(maxlen=self.n_max) # Bounded memory
        self.step = 0
        self.known_vocabulary = set()
        return self

    def _get_log_scaling_base(self) -> float:
        if not self.alphabet_autoscale:
            return 0.69314718056 # log(2)
        return math.log(max(2, len(self.known_vocabulary)))

    def fit(self, X: Union[Iterable[Any], Iterable[Iterable[Any]]], verbose: bool = True):
        """
        Fits on data. Handles both single stream and batch of sequences (List[List]).
        """
        # Batch detection heuristic
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
                self.reset_context() # Clear history between batch items
                for token in sequence:
                    self.update(token)
        else:
            for token in iterator:
                self.update(token)
        return self

    def predict_proba(self) -> Dict[Any, float]:
        if not self.history:
            return {}

        candidate_log_scores = defaultdict(lambda: -float('inf'))
        log_scale_base = self._get_log_scaling_base()
        
        # Cache locals for loop speed
        log_decay_val = self.log_decay
        current_step = self.step
        history_tuple = tuple(self.history)
        hist_len = len(history_tuple)
        found_pattern = False
        
        # Context Mixing: Scan suffixes from n_min to n_max
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
                # Math: log(Weight) = log(Count) + delta*log(Decay) + Length*log(Base)
                delta = current_step - node.last_visit_step
                node_factor = (delta * log_decay_val) + (length * log_scale_base)
                
                for token, count in node.counts.items():
                    if count <= 1e-9: continue
                    found_pattern = True
                    
                    log_weight = math.log(count) + node_factor
                    
                    # Manual inline of log_add_exp for performance
                    curr = candidate_log_scores[token]
                    if curr == -float('inf'):
                        candidate_log_scores[token] = log_weight
                    else:
                        if curr > log_weight:
                            candidate_log_scores[token] = curr + math.log1p(math.exp(log_weight - curr))
                        else:
                            candidate_log_scores[token] = log_weight + math.log1p(math.exp(curr - log_weight))

        if not found_pattern:
            if not self.known_vocabulary: return {}
            prob = 1.0 / len(self.known_vocabulary)
            return {k: prob for k in self.known_vocabulary}

        # Softmax Normalization (Shift trick)
        max_log = max(candidate_log_scores.values())
        linear_scores = {}
        total_sum = 0.0
        
        for token, log_score in candidate_log_scores.items():
            val = math.exp(log_score - max_log)
            linear_scores[token] = val
            total_sum += val
            
        return {t: v/total_sum for t, v in sorted(linear_scores.items(), key=lambda x: x[1], reverse=True)}

    def predict(self) -> Optional[Any]:
        probas = self.predict_proba()
        return next(iter(probas)) if probas else None

    def update(self, actual: Any):
        self.step += 1
        current_step = self.step
        decay_val = self.decay
        self.known_vocabulary.add(actual)
        
        history_tuple = tuple(self.history)
        hist_len = len(history_tuple)
        root = self.root
        
        # Update Trie paths
        # We always update starting from length 1 to build connectivity, regardless of n_min
        for length in range(1, self.n_max + 1):
            if hist_len < length: break
            
            context = history_tuple[-length:]
            node = root
            for token in context:
                if token not in node.children:
                    node.children[token] = TreeMemoryNode()
                node = node.children[token]
            
            # Inline Lazy Decay logic
            if node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = decay_val ** delta
                    keys_to_remove = []
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < 1e-5: keys_to_remove.append(t)
                        else: node.counts[t] = new_val
                    for t in keys_to_remove: del node.counts[t]
            
            node.last_visit_step = current_step
            node.counts[actual] += 1.0
            
        self.history.append(actual)

    # --- Context & Persistence ---

    def update_context(self, token: Any):
        """Append token to history without learning (Inference)."""
        self.history.append(token)

    def fill_context(self, context: Iterable[Any]):
        """Replace history with new sequence (Prompting)."""
        self.history.clear()
        self.history.extend(context)

    def reset_context(self):
        self.history.clear()

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f: pickle.dump(self, f)
        except Exception as e: print(f"Error saving: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'TreeMemoryPredictor':
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"Error loading: {e}"); return None