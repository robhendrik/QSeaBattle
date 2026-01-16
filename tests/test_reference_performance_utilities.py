import math
import pytest
import sys
sys.path.append("./src")

@pytest.mark.usefixtures("qsb")
def test_binary_entropy_basic_identities():
    from Q_Sea_Battle.reference_performance_utilities import binary_entropy

    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == pytest.approx(0.0)
    assert binary_entropy(1.0) == pytest.approx(0.0)

@pytest.mark.usefixtures("qsb")
def test_binary_entropy_invalid_domain_behavior():
    from Q_Sea_Battle.reference_performance_utilities import binary_entropy

    # Hard invalids should either raise OR return NaN (depending on implementation),
    # but tiny numerical drift is often tolerated by clipping.
    tiny_neg = -1e-9
    tiny_pos = 1.0 + 1e-9

    def call(p):
        return binary_entropy(p)

    for p in (tiny_neg, tiny_pos):
        try:
            v = call(p)
        except ValueError:
            # strict domain checking is fine
            continue

        # If it didn't raise, it should behave like a clipped probability
        # and produce a finite entropy close to 0 (since p ~= 0 or 1).
        assert isinstance(v, (float, int))
        assert math.isfinite(v)
        assert v >= -1e-12
        assert v <= 1e-6



@pytest.mark.usefixtures("qsb")
def test_binary_entropy_reverse_inverts_entropy():
    from Q_Sea_Battle.reference_performance_utilities import binary_entropy, binary_entropy_reverse

    for p in [0.1, 0.2, 0.3, 0.4]:
        h = binary_entropy(p)
        p_back = binary_entropy_reverse(h)
        # inverse is typically ambiguous (p and 1-p). We accept either.
        assert min(abs(p_back - p), abs(p_back - (1 - p))) < 1e-6


@pytest.mark.usefixtures("qsb")
def test_limit_from_mutual_information_known_hook():
    from Q_Sea_Battle.reference_performance_utilities import limit_from_mutual_information

    assert limit_from_mutual_information(field_size=4, comms_size=0) == pytest.approx(0.5)


@pytest.mark.usefixtures("qsb")
def test_expected_win_rate_functions_return_probabilities():
    from Q_Sea_Battle.reference_performance_utilities import (
        expected_win_rate_simple,
        expected_win_rate_majority,
        expected_win_rate_assisted,
    )

    for fn in [expected_win_rate_simple, expected_win_rate_majority, expected_win_rate_assisted]:
        p = fn(field_size=4, comms_size=1, enemy_probability=0.5, channel_noise=0.0)
        assert isinstance(p, (float, int))
        assert 0.0 <= float(p) <= 1.0


@pytest.mark.usefixtures("qsb")
def test_expected_win_rate_invalid_domains_raise():
    from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_simple

    with pytest.raises(ValueError):
        expected_win_rate_simple(field_size=0, comms_size=1)
    with pytest.raises(ValueError):
        expected_win_rate_simple(field_size=4, comms_size=-1)
    with pytest.raises(ValueError):
        expected_win_rate_simple(field_size=4, comms_size=1, enemy_probability=-0.1)
    with pytest.raises(ValueError):
        expected_win_rate_simple(field_size=4, comms_size=1, channel_noise=1.1)
