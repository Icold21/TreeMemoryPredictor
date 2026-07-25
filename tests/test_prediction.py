import pytest
from tree_memory_predictor import TreeMemoryPredictor

def test_basic_update_and_predict(empty_model):
    model = empty_model
    model.update("a")
    model.update("b")
    model.update("a")
    
    probas = model.predict_proba()
    assert "b" in probas
    assert probas["b"] > 0.0

def test_invalid_tokens(empty_model):
    model = empty_model
    with pytest.raises(TypeError):
        model.update(3.14)  # Float не разрешен
    with pytest.raises(TypeError):
        model.update(True)  # Bool не разрешен

def test_masked_modes(trained_model):
    for mode in ["none", "linear", "squared"]:
        probas = trained_model.predict_proba(masked_mode=mode)
        assert isinstance(probas, dict)