import pytest
from tree_memory_predictor import TreeMemoryPredictor

@pytest.fixture
def empty_model():
    return TreeMemoryPredictor(n_max=5, decay=0.9)

@pytest.fixture
def trained_model():
    model = TreeMemoryPredictor(n_max=5, decay=0.95)
    model.fit(["click", "buy", "click", "buy", "click", "buy", "exit"], verbose=False)
    return model