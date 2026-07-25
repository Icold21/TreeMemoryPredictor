import os
import pytest
from tree_memory_predictor import TreeMemoryPredictor

def test_json_serialization_deserialization(trained_model, tmp_path):
    filepath = os.path.join(tmp_path, "model.json")
    
    trained_model.save_json(filepath)
    assert os.path.exists(filepath)
    
    loaded_model = TreeMemoryPredictor.load_json(filepath)
    
    assert loaded_model.step == trained_model.step
    assert loaded_model.n_max == trained_model.n_max
    assert loaded_model.known_vocabulary == trained_model.known_vocabulary
    assert list(loaded_model.buffer._deque) == list(trained_model.buffer._deque)

def test_pickle_serialization_deserialization(trained_model, tmp_path):
    filepath = os.path.join(tmp_path, "model.pkl")
    
    trained_model.save(filepath)
    assert os.path.exists(filepath)
    
    loaded_model = TreeMemoryPredictor.load(filepath)
    assert loaded_model is not None
    assert loaded_model.step == trained_model.step