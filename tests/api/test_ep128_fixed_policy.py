import pytest

pytest.importorskip("torch")

from deep_ep.backends.direct.udma_policy import UdmaPolicy, resolve_udma_policy


@pytest.mark.parametrize('aiv', [0, 16, 32, 64])
@pytest.mark.parametrize('hidden', [2816, 6144, 7168])
@pytest.mark.parametrize('operation', ['dispatch', 'combine', 'weighted'])
def test_ep128_fixed_schedule(aiv, hidden, operation):
    policy = UdmaPolicy(128, 8, 1, 1, 16, ranks_per_group=16, topology_known=True)
    plan = policy.for_launch(aiv, hidden * 2, operation, bf16=True)
    assert plan.native_plan() == [1, 0, 16, 8, 1, 1, 16, 0, 0]
    assert plan.send_workers == 16
    assert policy.native_plan() == [1, 0, 16, 8, 1, 1, 16, 0, 0]


def test_ep128_low_core_capacity_fallback():
    policy = UdmaPolicy(128, 8, 1, 1, 16, ranks_per_group=16, topology_known=True)
    for operation in ('dispatch', 'combine', 'weighted'):
        plan = policy.for_launch(8, 5632, operation, bf16=True)
        assert plan.send_workers <= 8
        assert plan.valid_group_capacity()


def test_ep128_constructor_defaults_match_launch():
    policy = resolve_udma_policy(True, 128, 'udma', None, None, None, None, None, None, None)
    assert (policy.peers_per_worker, policy.wqes_per_visit) == (8, 16)


@pytest.mark.parametrize('ep', [8, 16, 32, 64])
def test_other_ep_base_defaults_unchanged(ep):
    policy = resolve_udma_policy(True, ep, 'udma', None, None, None, None, None, None, None)
    assert (policy.peers_per_worker, policy.wqes_per_visit) == ((1, 8) if ep <= 16 else (4, 32))
