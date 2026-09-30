import pytest

from critical_speed import critical_speed_m_per_s


def test_nominal_critical_speed() -> None:
    assert critical_speed_m_per_s(200.0) == pytest.approx(50.0)


@pytest.mark.parametrize("tension_n", [0.0, -1.0, float("inf"), float("nan")])
def test_critical_speed_rejects_invalid_tension(tension_n: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        critical_speed_m_per_s(tension_n)