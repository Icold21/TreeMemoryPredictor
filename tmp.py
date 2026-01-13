import math
import pickle
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any, Iterable

# Optional tqdm support for progress bars
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

class TreeMemoryNode:
    """
    Trie Node with __slots__ to significantly reduce memory footprint.
    """
    __slots__ = ['counts', 'children', 'last_visit_step']

    def __init__(self):
        self.counts = defaultdict(float) # Linear counts for O(1) updates
        self.children = {}
        self.last_visit_step = 0

class TreeMemoryPredictor:
    """
    Log-Space Context Mixing Model.
    Uses Entropy Scaling (S^L) and Lazy Exponential Decay.
    """
    def __init__(self, n_max: int = 10, decay: float = 0.99, alphabet_autoscale: bool = True):
        self.n_max = n_max
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        
        # Pre-calculate log decay to avoid repeated math.log() calls
        self.log_decay = math.log(self.decay) if self.decay > 0 else -float('inf')
        
        self.reset()

    def reset(self):
        self.root = TreeMemoryNode()
        self.history = deque(maxlen=self.n_max)
        self.step = 0
        self.known_vocabulary = set()
        return self

    def _get_log_scaling_base(self) -> float:
        """Returns log(VocabularySize) for entropy scaling."""
        if not self.alphabet_autoscale:
            return 0.69314718056 # math.log(2)
        return math.log(max(2, len(self.known_vocabulary)))

    def fit(self, X: Iterable[Any], verbose: bool = True):
        """
        Batch training.
        :param verbose: If True and tqdm is installed, shows progress bar.
        """
        iterator = X
        if verbose and _tqdm:
            total = len(X) if hasattr(X, '__len__') else None
            iterator = _tqdm(X, total=total, desc="TMP Fitting", unit="tok")
            
        for token in iterator:
            self.update(token)
        return self

    def predict_proba(self) -> Dict[Any, float]:
        if not self.history:
            return {}

        candidate_log_scores = defaultdict(lambda: -float('inf'))
        log_scale_base = self._get_log_scaling_base()
        
        # Cache local variables for speed in loops
        log_decay_val = self.log_decay
        current_step = self.step
        root = self.root
        history_tuple = tuple(self.history)
        hist_len = len(history_tuple)
        
        found_pattern = False
        
        # Context Mixing: Scan all suffixes up to n_max
        for length in range(1, self.n_max + 1):
            if hist_len < length: break
            
            # Fast slicing of tuple
            context = history_tuple[-length:]
            
            node = root
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
                    
                    # Manual inline of log_add_exp (a + log1p(exp(b-a))) for speed
                    current_score = candidate_log_scores[token]
                    if current_score == -float('inf'):
                        candidate_log_scores[token] = log_weight
                    else:
                        if current_score > log_weight:
                            candidate_log_scores[token] = current_score + math.log1p(math.exp(log_weight - current_score))
                        else:
                            candidate_log_scores[token] = log_weight + math.log1p(math.exp(current_score - log_weight))

        # Fallback: Uniform distribution
        if not found_pattern:
            if not self.known_vocabulary: return {}
            prob = 1.0 / len(self.known_vocabulary)
            return {k: prob for k in self.known_vocabulary}

        # Softmax Normalization (Shift trick for stability)
        max_log = max(candidate_log_scores.values())
        linear_scores = {}
        total_sum = 0.0
        
        for token, log_score in candidate_log_scores.items():
            val = math.exp(log_score - max_log)
            linear_scores[token] = val
            total_sum += val
            
        return {
            t: v / total_sum 
            for t, v in sorted(linear_scores.items(), key=lambda x: x[1], reverse=True)
        }

    def predict(self) -> Optional[Any]:
        probas = self.predict_proba()
        if not probas: return None
        return next(iter(probas))

    def update(self, actual: Any):
        """
        Updates the model. Contains inlined decay logic for max performance.
        """
        self.step += 1
        current_step = self.step
        decay_val = self.decay
        
        self.known_vocabulary.add(actual)
        
        history_tuple = tuple(self.history)
        hist_len = len(history_tuple)
        root = self.root
        
        # Update Trie for all suffix lengths
        for length in range(1, self.n_max + 1):
            if hist_len < length: break
            
            context = history_tuple[-length:]
            
            node = root
            for token in context:
                if token not in node.children:
                    node.children[token] = TreeMemoryNode()
                node = node.children[token]
            
            # --- INLINED LAZY DECAY ---
            # Apply decay only when visiting the node
            if node.last_visit_step != 0:
                delta = current_step - node.last_visit_step
                if delta > 0:
                    factor = decay_val ** delta
                    
                    # Apply factor to all counts and prune small values
                    # Using list() to allow modification during iteration
                    keys_to_remove = []
                    for t, c in node.counts.items():
                        new_val = c * factor
                        if new_val < 1e-5:
                            keys_to_remove.append(t)
                        else:
                            node.counts[t] = new_val
                    
                    for t in keys_to_remove:
                        del node.counts[t]
            
            node.last_visit_step = current_step
            # --------------------------
            
            node.counts[actual] += 1.0
            
        self.history.append(actual)

    # --- Context Management ---

    def update_context(self, token: Any):
        """Appends token to history without updating weights (Inference mode)."""
        self.history.append(token)

    def fill_context(self, context: Iterable[Any]):
        """Replaces history with a new sequence (Prompting)."""
        self.history.clear()
        self.history.extend(context)

    def reset_context(self):
        self.history.clear()

    # --- Persistence ---

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
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None