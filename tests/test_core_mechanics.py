import pytest
from tree_memory_predictor import TokenBuffer, TreeMemoryNode

def test_token_buffer_initialization():
    buf = TokenBuffer(maxlen=3)
    assert buf.size == 0
    assert buf.to_tuple() == ()

def test_token_buffer_append_and_overflow():
    buf = TokenBuffer(maxlen=3)
    buf.append(1)
    buf.append(2)
    assert buf.size == 2
    assert buf.to_tuple() == (1, 2)
    
    buf.append(3)
    buf.append(4)
    assert buf.size == 3
    assert buf.to_tuple() == (2, 3, 4)

def test_token_buffer_extend_and_clear():
    buf = TokenBuffer(maxlen=3)
    buf.extend([1, 2, 3, 4])
    assert buf.size == 3
    assert buf.to_tuple() == (2, 3, 4)
    
    buf.clear()
    assert buf.size == 0
    assert buf.to_tuple() == ()

def test_tree_memory_node_serialization():
    node = TreeMemoryNode()
    node.counts["a"] = 2.5
    node.last_visit_step = 42
    
    child = TreeMemoryNode()
    child.counts["b"] = 1.0
    node.children["a"] = child
    
    serialized = node.to_dict()
    restored = TreeMemoryNode.from_dict(serialized)
    
    assert restored.counts["a"] == 2.5
    assert restored.last_visit_step == 42
    assert "a" in restored.children
    assert restored.children["a"].counts["b"] == 1.0