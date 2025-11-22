from tender_analyzer.common.vectorstore.qdrant_client import text_to_vector


def test_text_to_vector_has_expected_length():
    vec = text_to_vector("safe Tender text", 32)
    assert len(vec) == 32


def test_text_to_vector_values_are_normalized():
    vec = text_to_vector("safe Tender text", 16)
    assert all(0.0 <= value <= 1.0 for value in vec)
