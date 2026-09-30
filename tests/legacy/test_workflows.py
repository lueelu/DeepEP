# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Pending V1 BF16 communication qualification; opt in with --run-device."""

import pytest

from tests.utils.cases import ROUTES, make_case
from tests.utils.device import buffer_for, run_cached, run_operation

pytestmark = pytest.mark.device


@pytest.mark.parametrize("name", ROUTES)
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("operation", ["dispatch", "combine", "roundtrip"])
def test_legacy(device_session, name, with_weights, operation):
    if device_session[2].world_size != 2:
        pytest.skip("legacy semantic scenarios use the two-rank hand-checkable routes")
    case = make_case(name, hidden=256, with_weights=with_weights)
    with buffer_for(device_session, case, "legacy") as buffer:
        run_operation(device_session, buffer, case, "legacy", False, operation)


def test_cached_dispatch(device_session):
    if device_session[2].world_size != 2:
        pytest.skip("legacy cached scenario uses the two-rank hand-checkable routes")
    case = make_case("normal", hidden=256, with_weights=False)
    with buffer_for(device_session, case, "legacy") as buffer:
        run_cached(device_session, buffer, case, "legacy", False)
