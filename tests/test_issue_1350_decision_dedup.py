"""Issue #1350 — an executed decision is not re-sent to a cover resting short of it.

A cover that cannot land on the exact commanded number (a Shelly reporting 9
after being sent 10) was re-commanded on every cycle whenever the delta gate
was bypassed — a special target (default, sunset, My, an always-enforced
limit), a move away from one, or a forced dispatch — because the same-position
gate compared the reading with exact equality. The gate now also skips when the
target is the decision already put on the wire and the cover rests within
``position_tolerance`` of its booked target.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.adaptive_cover_pro.const import CONF_DEFAULT_HEIGHT
from custom_components.adaptive_cover_pro.managers.cover_command import (
    CoverCommandService,
    PositionContext,
    build_special_positions,
)

ENTITY = "cover.test"
SENT = ("sent", "set_cover_position")
SKIPPED = ("skipped", "same_position")
_CAPS = {
    "has_set_position": True,
    "has_set_tilt_position": False,
    "has_open": True,
    "has_close": True,
}


@pytest.fixture
def hass():
    """Return a mock hass exposing one available, position-capable cover."""
    hass = MagicMock()
    state_obj = MagicMock()
    state_obj.state = "open"
    state_obj.attributes = {"current_position": 100, "supported_features": 15}
    hass.states.get.return_value = state_obj
    hass.services.async_call = AsyncMock(return_value=None)
    return hass


def _svc(hass, tolerance: int = 3) -> CoverCommandService:
    """Return a CoverCommandService for a vertical blind with *tolerance*."""
    return CoverCommandService(
        hass=hass,
        logger=MagicMock(),
        cover_type="cover_blind",
        grace_mgr=MagicMock(),
        position_tolerance=tolerance,
    )


def _ctx(*, special=None, min_change: int = 5, force: bool = False) -> PositionContext:
    """Return a context that passes every gate but same-position and delta."""
    return PositionContext(
        auto_control=True,
        manual_override=False,
        sun_just_appeared=False,
        min_change=min_change,
        time_threshold=0,
        special_positions=[0, 100] if special is None else special,
        force=force,
    )


class _Cover:
    """Drive ``apply_position`` against a settable reported position."""

    def __init__(self, svc: CoverCommandService, hass, position: int) -> None:
        self.svc = svc
        self.hass = hass
        self.position = position

    async def apply(self, target: int, ctx: PositionContext) -> tuple[str, str]:
        """Run one update cycle for *target* with the current reading."""
        with (
            patch.object(
                self.svc,
                "_get_current_position",
                side_effect=lambda _entity: self.position,
            ),
            patch.object(self.svc, "_check_time_delta", return_value=True),
            patch(
                "custom_components.adaptive_cover_pro.managers.cover_command.check_cover_features",
                return_value=_CAPS,
            ),
        ):
            return await self.svc.apply_position(ENTITY, target, "default", ctx)

    @property
    def sends(self) -> int:
        """Return how many service calls reached the cover."""
        return self.hass.services.async_call.call_count


@pytest.mark.asyncio
async def test_default_position_not_resent_when_cover_lands_one_short(hass):
    """The reported defect: default 10, the cover reports 9 after the move."""
    special = build_special_positions({CONF_DEFAULT_HEIGHT: 10})
    cover = _Cover(_svc(hass), hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    assert await cover.apply(10, _ctx(special=special)) == SKIPPED
    assert await cover.apply(10, _ctx(special=special)) == SKIPPED
    assert cover.sends == 1


@pytest.mark.asyncio
async def test_forced_dispatch_not_resent_when_cover_lands_one_short(hass):
    """A forced dispatch skips the delta gate; the decision check still holds it."""
    cover = _Cover(_svc(hass), hass, position=100)

    assert await cover.apply(30, _ctx(force=True)) == SENT
    cover.position = 29
    assert await cover.apply(30, _ctx(force=True)) == SKIPPED
    assert cover.sends == 1


@pytest.mark.asyncio
async def test_move_away_from_special_not_resent_when_cover_stays_put(hass):
    """Leaving a special bypasses the delta gate; a cover that stays is not chased."""
    special = [0, 10, 100]
    cover = _Cover(_svc(hass), hass, position=10)

    assert await cover.apply(12, _ctx(special=special)) == SENT
    assert await cover.apply(12, _ctx(special=special)) == SKIPPED
    assert cover.sends == 1


@pytest.mark.asyncio
async def test_new_decision_within_tolerance_is_still_sent(hass):
    """Issue #567 guard: a new target near the current reading reaches the delta gate."""
    cover = _Cover(_svc(hass), hass, position=100)

    assert await cover.apply(30, _ctx(min_change=1)) == SENT
    cover.position = 29
    assert await cover.apply(31, _ctx(min_change=1)) == SENT
    assert cover.sends == 2


@pytest.mark.asyncio
async def test_decision_resent_once_cover_leaves_tolerance(hass):
    """A cover moved out of the tolerance band is re-commanded to the same decision."""
    special = [0, 10, 100]
    cover = _Cover(_svc(hass), hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    assert await cover.apply(10, _ctx(special=special)) == SKIPPED
    cover.position = 20
    assert await cover.apply(10, _ctx(special=special)) == SENT
    assert cover.sends == 2


@pytest.mark.asyncio
async def test_manual_override_edge_forgets_decision(hass):
    """``discard_target`` (both override edges) drops the decision with the row."""
    special = [0, 10, 100]
    svc = _svc(hass)
    cover = _Cover(svc, hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    svc.discard_target(ENTITY)
    assert await cover.apply(10, _ctx(special=special)) == SENT
    assert cover.sends == 2


@pytest.mark.asyncio
async def test_carriage_rebase_keeps_decision(hass):
    """The dual-axis rebase moves ``target`` but not the decision behind it."""
    special = [0, 10, 100]
    svc = _svc(hass, tolerance=0)
    cover = _Cover(svc, hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    svc.rebase_target(ENTITY, 9)
    assert svc.state(ENTITY).decided == 10
    assert await cover.apply(10, _ctx(special=special)) == SKIPPED
    assert cover.sends == 1


@pytest.mark.asyncio
async def test_foreign_target_write_forgets_decision(hass):
    """Any other writer that moves ``target`` invalidates the decision."""
    special = [0, 10, 100]
    svc = _svc(hass)
    cover = _Cover(svc, hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    svc.set_target(ENTITY, 9)
    assert svc.state(ENTITY).decided is None
    assert await cover.apply(10, _ctx(special=special)) == SENT
    assert cover.sends == 2


@pytest.mark.asyncio
async def test_restating_the_same_target_keeps_decision(hass):
    """A reconciliation-style restatement of the booked value keeps the decision."""
    special = [0, 10, 100]
    svc = _svc(hass)
    cover = _Cover(svc, hass, position=100)

    assert await cover.apply(10, _ctx(special=special)) == SENT
    cover.position = 9
    svc.set_target(ENTITY, 10)
    assert await cover.apply(10, _ctx(special=special)) == SKIPPED
    assert cover.sends == 1
