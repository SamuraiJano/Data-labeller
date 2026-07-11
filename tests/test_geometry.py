from tools.dataset_labeler.geometry import (
    denormalize_polygon,
    is_self_intersecting,
    normalize_polygon,
    validate_polygon,
)


def test_polygon_coordinate_normalization_round_trip() -> None:
    points = [(10.0, 20.0), (50.0, 20.0), (50.0, 80.0), (10.0, 80.0)]
    normalized = normalize_polygon(points, width=100, height=200)
    assert normalized == [0.1, 0.1, 0.5, 0.1, 0.5, 0.4, 0.1, 0.4]
    assert denormalize_polygon(normalized, width=100, height=200) == points


def test_invalid_self_intersecting_polygon_detection() -> None:
    bow_tie = [(0.0, 0.0), (20.0, 20.0), (0.0, 20.0), (20.0, 0.0)]
    assert is_self_intersecting(bow_tie)
    assert any("self-intersecting" in error for error in validate_polygon(bow_tie, 30, 30))
