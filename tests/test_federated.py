import pytest
from tree_memory_predictor import TreeMemoryPredictor

def test_federated_merge():
    model_a = TreeMemoryPredictor(n_max=3, decay=0.9)
    model_b = TreeMemoryPredictor(n_max=5, decay=0.95)
    
    model_a.fit(["click", "buy", "click"], verbose=False)
    model_b.fit(["exit", "exit", "buy", "exit"], verbose=False)
    
    model_a.merge(model_b)
    
    assert model_a.n_max == 5
    assert "exit" in model_a.known_vocabulary
    assert "click" in model_a.known_vocabulary