from amplifier_data.doctor import report,self_test
from amplifier_data import RUST_AVAILABLE


def test_two_consumers_regenerate_without_native_requirement():
    result=self_test()
    assert result['fictional_consumers']==2
    assert result['shared_kernel'] and result['scoped_views'] and result['cell_regeneration']
    assert not result['durable_restart']


def test_doctor_never_claims_unsupported_persistence():
    result=report()
    assert result['durable_available']==RUST_AVAILABLE
    assert not result['owned_model'] and not result['network_calls']


def test_two_consumers_survive_restart_when_native_built():
    import pytest
    if not RUST_AVAILABLE:pytest.skip('Rust extension not built; no durability result claimed')
    assert self_test(True)['durable_restart']
