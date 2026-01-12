import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any

class TreeMemoryNode:
    """
    Trie Node allowing dynamic alphabet growth and count decay.
    """
    def __init__(self):
        # Stores counts as floats for decay: {'a': 10.5, 'b': 2.1}
        self.counts = defaultdict(float)
        self.children = {}
        self.last_visit_step = 0 # Initialize at 0 to sync with predictor step

class TreeMemoryPredictor:
    """
    A General Context-Mixing Predictor with adaptive Count Decay.
    
    Parameters
    ----------
    n_max : int
        Maximum depth of context to search.
        
    decay : float
        The factor by which old counts are multiplied at each step.
        Example: 0.99 means counts lose 1% of their value every step they are not visited.
        This allows the model to "change its mind" about established patterns.
        
    alphabet_autoscale : bool, default=True
        If True, weight predictions by S^L (Vocabulary Size ^ Length).
    """
    def __init__(self, n_max: int = 10, decay: float = 0.99, alphabet_autoscale: bool = True):
        self.n_max = n_max
        self.decay = decay
        self.alphabet_autoscale = alphabet_autoscale
        
        self.reset()

    def reset(self):
        """
        Clears all memory, history, and vocabulary. 
        Resets the model to its initial state.
        """
        self.root = TreeMemoryNode()
        self.history: List[Any] = []
        self.step = 0
        self.known_vocabulary = set()
        return self

    def _get_scaling_base(self) -> int:
        if not self.alphabet_autoscale:
            return 2
        vocab_size = len(self.known_vocabulary)
        return max(2, vocab_size)

    def _decay_node_counts(self, node: TreeMemoryNode, current_step: int):
        """
        Applies 'lazy decay' to the counts within a node based on how long 
        it has been since the last visit.
        """
        if node.last_visit_step == 0:
            # First visit, nothing to decay
            node.last_visit_step = current_step
            return

        delta = current_step - node.last_visit_step
        if delta > 0:
            factor = self.decay ** delta
            # Apply decay to all existing counts
            for token in list(node.counts.keys()):
                node.counts[token] *= factor
                # Optimization: Remove very small residuals to save memory
                if node.counts[token] < 1e-4:
                    del node.counts[token]
            
            node.last_visit_step = current_step

    def fit(self, X: List[Any]):
        for token in X:
            self.update(token)
        return self

    def predict_proba(self) -> Dict[Any, float]:
        if not self.history:
            return {}

        candidate_scores = defaultdict(float)
        total_weight_sum = 0.0
        
        scale_base = self._get_scaling_base()
        
        for length in range(1, self.n_max + 1):
            if len(self.history) < length:
                break
            
            context = tuple(self.history[-length:])
            
            # --- Trie Traversal ---
            node = self.root
            path_exists = True
            for token in context:
                if token not in node.children:
                    path_exists = False
                    break
                node = node.children[token]
            
            if path_exists:
                # Calculate what the total would be if we visited it now
                # (Temporary decay for prediction accuracy)
                delta = self.step - node.last_visit_step
                temp_decay = self.decay ** delta
                
                # Sum decayed counts
                current_total_obs = sum(v * temp_decay for v in node.counts.values())
                
                if current_total_obs < 1e-6: continue

                # Complexity Weight: Base^Length
                complexity_factor = float(scale_base ** length)
                
                # Weight = (Decayed Observations) * (Complexity)
                # Note: We use current_total_obs as the weight metric directly,
                # because it already contains the 'activity' information via decay.
                level_weight = current_total_obs * complexity_factor
                
                for token, count in node.counts.items():
                    # Probability using decayed values
                    decayed_count = count * temp_decay
                    local_prob = decayed_count / current_total_obs
                    
                    candidate_scores[token] += local_prob * level_weight
                
                total_weight_sum += level_weight

        if total_weight_sum == 0:
            if not self.known_vocabulary: return {}
            uniform_prob = 1.0 / len(self.known_vocabulary)
            return {k: uniform_prob for k in self.known_vocabulary}

        result = {
            token: score / total_weight_sum 
            for token, score in candidate_scores.items()
        }
        
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def predict(self) -> Optional[Any]:
        probas = self.predict_proba()
        if not probas: return None
        return max(probas, key=probas.get)

    def update(self, actual: Any):
        self.step += 1
        self.known_vocabulary.add(actual)
        
        for length in range(1, self.n_max + 1):
            if len(self.history) < length:
                break
            
            context = tuple(self.history[-length:])
            
            node = self.root
            for token in context:
                if token not in node.children:
                    node.children[token] = TreeMemoryNode()
                node = node.children[token]
            
            # --- CRITICAL: Apply Decay to Counts ---
            # Before adding the new observation, we shrink the old ones.
            self._decay_node_counts(node, self.step)
            
            # Add new observation (strength 1.0)
            node.counts[actual] += 1.0
            
        self.history.append(actual)