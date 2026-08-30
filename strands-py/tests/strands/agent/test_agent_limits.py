"""Tests for per-invocation budget caps via the ``limits`` kwarg."""

from collections.abc import Sequence

import pytest

from strands import Agent, tool
from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent
from strands.types.event_loop import Usage
from tests.fixtures.mocked_model_provider import MockedModelProvider

TOOL_USE_RESPONSE_1 = {
    "role": "assistant",
    "content": [{"toolUse": {"toolUseId": "tool-1", "name": "loop", "input": {}}}],
}
TOOL_USE_RESPONSE_2 = {
    "role": "assistant",
    "content": [{"toolUse": {"toolUseId": "tool-2", "name": "loop", "input": {}}}],
}


@tool
def loop() -> str:
    """A passthrough tool used to drive multi-turn loops in tests."""
    return "ok"


def _usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(
        inputTokens=input_tokens,
        outputTokens=output_tokens,
        totalTokens=input_tokens + output_tokens,
    )


def _agent(responses: Sequence[dict], usages: Sequence[Usage] | None = None) -> Agent:
    return Agent(
        model=MockedModelProvider(list(responses), list(usages) if usages is not None else None),
        tools=[loop],
    )


@pytest.mark.asyncio
async def test_limit_turns_bails_after_cycle_completes():
    """With turns=1, the first tool round trip runs but the second model call is skipped."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(20, 5)],
    )

    result = await agent.invoke_async("go", limits={"turns": 1})

    assert result.stop_reason == "limit_turns"
    # Tools from cycle 1 ran; cycle 2 was skipped, so only one model call's usage is recorded.
    assert result.metrics.latest_agent_invocation is not None
    assert len(result.metrics.latest_agent_invocation.cycles) == 1
    assert result.metrics.latest_agent_invocation.usage["outputTokens"] == 5
    # lastMessage should be the user toolResult (tools were processed before the cap fired).
    assert result.message["role"] == "user"
    assert any("toolResult" in block for block in result.message["content"])


@pytest.mark.asyncio
async def test_generous_limits_does_not_trip():
    final_response = {"role": "assistant", "content": [{"text": "done"}]}
    agent = Agent(model=MockedModelProvider([final_response], [_usage(5, 5)]))

    result = await agent.invoke_async("go", limits={"turns": 5, "output_tokens": 1000, "total_tokens": 1000})

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_limit_output_tokens_trips_when_cumulative_meets_cap():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 60), _usage(10, 60)],
    )

    result = await agent.invoke_async("go", limits={"output_tokens": 100})

    assert result.stop_reason == "limit_output_tokens"
    assert result.metrics.latest_agent_invocation is not None
    assert len(result.metrics.latest_agent_invocation.cycles) == 2


@pytest.mark.asyncio
async def test_limit_output_tokens_uses_at_least_semantics():
    """Stops when count exactly equals the cap (>= semantics)."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 100), _usage(10, 100)],
    )

    result = await agent.invoke_async("go", limits={"output_tokens": 100})

    assert result.stop_reason == "limit_output_tokens"
    assert len(result.metrics.latest_agent_invocation.cycles) == 1


@pytest.mark.asyncio
async def test_limit_total_tokens_trips_when_cumulative_meets_cap():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(200, 100), _usage(200, 100)],
    )

    result = await agent.invoke_async("go", limits={"total_tokens": 500})

    assert result.stop_reason == "limit_total_tokens"
    assert len(result.metrics.latest_agent_invocation.cycles) == 2


@pytest.mark.asyncio
async def test_end_turn_wins_when_model_finishes_on_the_same_cycle_a_cap_would_trip():
    final = {"role": "assistant", "content": [{"text": "final answer"}]}
    agent = Agent(model=MockedModelProvider([final], [_usage(300, 300)]))

    result = await agent.invoke_async("go", limits={"total_tokens": 500})

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_simultaneous_trip_priority_turns_wins():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(100, 100), _usage(100, 100)],
    )

    result = await agent.invoke_async("go", limits={"turns": 1, "total_tokens": 1, "output_tokens": 1})

    assert result.stop_reason == "limit_turns"


@pytest.mark.asyncio
async def test_simultaneous_trip_priority_total_over_output():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(100, 100), _usage(100, 100)],
    )

    result = await agent.invoke_async("go", limits={"total_tokens": 1, "output_tokens": 1})

    assert result.stop_reason == "limit_total_tokens"


@pytest.mark.asyncio
async def test_falls_back_to_output_tokens_when_no_higher_priority_cap_set():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(100, 100), _usage(100, 100)],
    )

    result = await agent.invoke_async("go", limits={"output_tokens": 1})

    assert result.stop_reason == "limit_output_tokens"


@pytest.mark.asyncio
async def test_limits_are_per_invocation_not_lifetime():
    """A reused agent gets a fresh budget on each invoke; the cap doesn't leak across calls."""
    final = {"role": "assistant", "content": [{"text": "done"}]}
    agent = Agent(model=MockedModelProvider([final, final], [_usage(10, 50), _usage(10, 50)]))

    r1 = await agent.invoke_async("go", limits={"output_tokens": 75})
    assert r1.stop_reason == "end_turn"
    assert len(r1.metrics.latest_agent_invocation.cycles) == 1

    r2 = await agent.invoke_async("go again", limits={"output_tokens": 75})
    assert r2.stop_reason == "end_turn"
    assert len(r2.metrics.latest_agent_invocation.cycles) == 1


@pytest.mark.parametrize(
    "limits",
    [
        {"turns": -1},
        {"turns": 0},
        {"output_tokens": 0},
        {"total_tokens": -5},
        {"turns": True},  # bool subclasses int; should still be rejected
    ],
)
@pytest.mark.asyncio
async def test_invalid_limits_raise_type_error(limits):
    final = {"role": "assistant", "content": [{"text": "never reached"}]}
    agent = Agent(model=MockedModelProvider([final]))

    with pytest.raises(TypeError):
        await agent.invoke_async("go", limits=limits)


@pytest.mark.asyncio
async def test_limits_propagate_through_stream_async():
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )

    result = None
    async for event in agent.stream_async("go", limits={"turns": 1}):
        if "result" in event:
            result = event["result"]

    assert result is not None
    assert result.stop_reason == "limit_turns"


def test_limits_propagate_through_call():
    """The sync ``__call__`` path also accepts and applies ``limits``."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )

    result = agent("go", limits={"turns": 1})

    assert result.stop_reason == "limit_turns"


@pytest.mark.asyncio
async def test_empty_limits_dict_is_no_op():
    """``limits={}`` carries no caps and behaves identically to ``limits=None``."""
    final = {"role": "assistant", "content": [{"text": "done"}]}
    agent = Agent(model=MockedModelProvider([final], [_usage(5, 5)]))

    result = await agent.invoke_async("go", limits={})

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_invocation_hooks_fire_when_limit_trips():
    """``BeforeInvocationEvent`` / ``AfterInvocationEvent`` fire even on a graceful cap-trip."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )

    seen: list[str] = []
    agent.add_hook(lambda event: seen.append("before"), BeforeInvocationEvent)
    agent.add_hook(lambda event: seen.append("after"), AfterInvocationEvent)

    result = await agent.invoke_async("go", limits={"turns": 1})

    assert result.stop_reason == "limit_turns"
    assert seen == ["before", "after"]


# ---------------------------------------------------------------------------
# Constructor-level ``max_iterations`` — issue #2298
#
# ``Agent(max_iterations=N)`` applies ``{"turns": N}`` as a default turn cap on every
# invocation that does not pass its own ``limits`` arg (or that passes a limits dict
# without a ``turns`` key). Callers can extend (add more caps) or override (set
# ``turns`` explicitly). Trip behavior matches per-invocation: ``stop_reason="limit_turns"``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_constructor_applies_default_turns_cap():
    """``Agent(max_iterations=1)`` makes every invocation use ``limits={"turns": 1}``."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )

    capped = Agent(
        model=agent.model,
        tools=[loop],
        max_iterations=1,
    )

    result = await capped.invoke_async("go")

    assert result.stop_reason == "limit_turns"
    assert result.metrics.latest_agent_invocation is not None
    assert len(result.metrics.latest_agent_invocation.cycles) == 1


@pytest.mark.asyncio
async def test_max_iterations_constructor_is_default_not_locked_in():
    """A caller that sets ``turns`` explicitly in ``limits`` overrides the constructor cap."""
    final_response = {"role": "assistant", "content": [{"text": "done"}]}
    agent = Agent(
        model=MockedModelProvider(
            [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2, final_response],
            [_usage(10, 5), _usage(10, 5), _usage(10, 5)],
        ),
        tools=[loop],
        max_iterations=1,  # would trip on its own at cycle 1
    )

    # Caller overrides with ``turns=5``: both tool-use cycles run, then the model returns final.
    result = await agent.invoke_async("go", limits={"turns": 5})

    assert result.stop_reason == "end_turn"
    assert len(result.metrics.latest_agent_invocation.cycles) == 3


@pytest.mark.asyncio
async def test_max_iterations_constructor_merges_with_caller_output_tokens_cap():
    """Constructor ``max_iterations`` applies even when the caller sets a different cap."""
    final_response = {"role": "assistant", "content": [{"text": "done"}]}
    agent = Agent(
        model=MockedModelProvider([final_response], [_usage(10, 50)]),
        max_iterations=5,  # would NOT trip on its own; caller adds output_tokens cap
    )

    result = await agent.invoke_async("go", limits={"output_tokens": 100})

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_max_iterations_none_preserves_unbounded_behavior():
    """Default ``max_iterations=None`` (no constructor arg) does not impose any cap."""
    # 100 staged tool-use responses so an unintended cap would fire well before the end.
    responses: list[dict] = [{"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t-{i}", "name": "loop", "input": {}}}]} for i in range(20)]
    usages = [_usage(5, 5)] * 20
    # Add a final non-tool response so the loop terminates cleanly once the tool calls exhaust.
    responses.append({"role": "assistant", "content": [{"text": "done"}]})
    usages.append(_usage(5, 5))

    agent = Agent(
        model=MockedModelProvider(responses, usages),
        tools=[loop],
    )

    result = await agent.invoke_async("go")

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_max_iterations_propagates_through_stream_async():
    """Constructor cap applies to the streaming entry point too."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )
    capped = Agent(
        model=agent.model,
        tools=[loop],
        max_iterations=1,
    )

    result = None
    async for event in capped.stream_async("go"):
        if "result" in event:
            result = event["result"]

    assert result is not None
    assert result.stop_reason == "limit_turns"


def test_max_iterations_propagates_through_call():
    """The sync ``__call__`` path also inherits the constructor cap."""
    agent = _agent(
        [TOOL_USE_RESPONSE_1, TOOL_USE_RESPONSE_2],
        usages=[_usage(10, 5), _usage(10, 5)],
    )
    capped = Agent(
        model=agent.model,
        tools=[loop],
        max_iterations=1,
    )

    result = capped("go")

    assert result.stop_reason == "limit_turns"


@pytest.mark.parametrize(
    "bad_value",
    [0, -1, -100, True, False, 1.5, "5"],
)
def test_max_iterations_invalid_raises_type_error(bad_value):
    """Bad constructor values fail fast with ``TypeError`` (same shape as ``_validate_limits``)."""
    with pytest.raises(TypeError, match="max_iterations must be a positive int"):
        Agent(model=MockedModelProvider([]), max_iterations=bad_value)


def test_max_iterations_valid_positive_int_accepted():
    """Sanity: any positive int is accepted; the cap is stored as-is."""
    agent = Agent(model=MockedModelProvider([]), max_iterations=42)

    assert agent._max_iterations == 42
