"""Regression coverage for the shared nested state-body declaration reader.

These tests describe the public declaration contract from issue #657.  They
exercise observable graph, event, callback, and inheritance behavior rather
than private implementation details.
"""

from enum import Enum

import pytest
from statemachine.states import States

from statemachine import Event
from statemachine import State
from statemachine import StateChart


def assert_parallel_leaf_values(chart_cls):
    expected = {"left-idle", "left-active", "right-idle", "right-active"}
    assert expected <= set(chart_cls.states_map)


def test_nested_event_forms_keep_aliases_and_metadata():
    class Journey(StateChart):
        class route(State.Compound):
            start = State(initial=True)
            finish = State(final=True)
            arrive = start.to(finish)
            explicit = Event(arrive, id="route.arrive", name="Arrive", internal=True)
            floating = Event(id="route.ping", name="Ping", delay=0.25, internal=True)

            @start.to(finish)
            def revisit(self):
                return None

        done = State(final=True)
        complete = route.to(done)

    assert Journey.explicit.id == "route.arrive"
    assert Journey.explicit.name == "Arrive"
    assert Journey.explicit.internal is True
    assert Journey.floating.id == "route.ping"
    assert Journey.floating.name == "Ping"
    assert Journey.floating.delay == pytest.approx(0.25)
    assert Journey.floating.internal is True
    assert {"route.arrive", "route.ping"} <= {event.id for event in Journey.events}
    assert Journey.explicit is not None
    assert Journey.revisit is not None


def test_nested_alias_families_and_enum_states_remain_usable():
    class Mode(Enum):
        IDLE = 1
        DONE = 2

    class Workflow(StateChart):
        class process(State.Compound):
            states = States.from_enum(Mode, initial=Mode.IDLE, final=Mode.DONE)
            begin = states.IDLE.to(states.DONE)

            error_execution = states.IDLE.to(states.DONE)
            done_state_process = states.IDLE.to(states.DONE)
            done_invoke_process = states.IDLE.to(states.DONE)

        finished = State(final=True)
        leave = process.to(finished)

    ids = {event.id for event in Workflow.events}
    assert "error.execution" in ids
    assert "done.state.process" in ids
    assert "done.invoke.process" in ids
    assert Mode.IDLE in Workflow.states_map
    assert Mode.DONE in Workflow.states_map


@pytest.mark.asyncio()
async def test_parallel_same_named_callbacks_keep_owner_scope(sm_runner):
    calls = []

    class ParallelChart(StateChart):
        class branches(State.Parallel):
            class left(State.Compound):
                idle = State(name="Left idle", value="left-idle", initial=True)
                active = State(name="Left active", value="left-active")
                go = idle.to(active)

                def on_enter_active(self):
                    calls.append("left")

            class right(State.Compound):
                idle = State(name="Right idle", value="right-idle", initial=True)
                active = State(name="Right active", value="right-active")
                go = idle.to(active)

                def on_enter_active(self):
                    calls.append("right")

        start = State(initial=True)
        enter = start.to(branches)

    assert_parallel_leaf_values(ParallelChart)
    sm = await sm_runner.start(ParallelChart)
    await sm_runner.send(sm, "enter")
    await sm_runner.send(sm, "go")

    assert sorted(calls) == ["left", "right"]


@pytest.mark.asyncio()
async def test_parallel_callable_callbacks_keep_owner_scope(sm_runner):
    calls = []

    class ParallelChart(StateChart):
        class branches(State.Parallel):
            class left(State.Compound):
                idle = State(name="Left idle", value="left-idle", initial=True)

                def enter_shared(self):
                    calls.append("left-callable")

                active = State(name="Left active", value="left-active", enter=enter_shared)
                go = idle.to(active)

            class right(State.Compound):
                idle = State(name="Right idle", value="right-idle", initial=True)

                def enter_shared(self):
                    calls.append("right-callable")

                active = State(name="Right active", value="right-active", enter=enter_shared)
                go = idle.to(active)

        start = State(initial=True)
        enter = start.to(branches)

    assert_parallel_leaf_values(ParallelChart)
    sm = await sm_runner.start(ParallelChart)
    await sm_runner.send(sm, "enter")
    await sm_runner.send(sm, "go")

    assert sorted(calls) == ["left-callable", "right-callable"]


def test_nested_callbacks_are_reusable_across_base_child_and_sibling():
    calls = []

    class Base(StateChart):
        class flow(State.Compound):
            idle = State(initial=True)
            done = State()
            finish = idle.to(done)

            def on_enter_done(self):
                calls.append("base")

        stop = State(final=True)
        leave = flow.to(stop)

    class Child(Base):
        pass

    class Sibling(Base):
        pass

    Base().send("finish")
    Child().send("finish")
    Sibling().send("finish")

    assert calls == ["base", "base", "base"]


def test_nested_guard_reads_live_machine_property():
    class Model:
        allow = False

    class Guarded(StateChart):
        class flow(State.Compound):
            waiting = State(initial=True)
            accepted = State()
            property_accepted = State()
            constant_accepted = State(final=True)
            constant_allowed = False

            @property
            def allowed(self):
                return self.enabled

            @property
            def direct_allowed(self):
                return self.enabled

            approve = waiting.to(accepted, cond="allowed")
            approve_property = accepted.to(property_accepted, cond=direct_allowed)
            approve_constant = property_accepted.to(constant_accepted, cond="constant_allowed")
            approve_expression = property_accepted.to(
                constant_accepted, cond="direct_allowed and allow"
            )

        # The local false value must win over the outer true value.
        constant_allowed = True

        def __init__(self, model=None):
            self.enabled = False
            super().__init__(model=model)

    model = Model()
    machine = Guarded(model=model)
    machine.send("approve")
    assert set(machine.configuration_values) == {"flow", "waiting"}

    machine.enabled = True
    machine.send("approve")
    assert set(machine.configuration_values) == {"flow", "accepted"}
    machine.send("approve_property")
    assert set(machine.configuration_values) == {"flow", "property_accepted"}
    machine.send("approve_constant")
    assert set(machine.configuration_values) == {"flow", "property_accepted"}
    model.allow = True
    machine.send("approve_expression")
    assert set(machine.configuration_values) == {"flow", "constant_accepted"}


def test_machine_model_and_listener_callbacks_coexist_in_declared_order():
    calls = []

    class Model:
        def on_enter_ready(self):
            calls.append("model")

    class Chart(StateChart):
        start = State(initial=True)
        ready = State(final=True)
        go = start.to(ready)

        def on_enter_ready(self):
            calls.append("machine")

    class Listener:
        def on_enter_ready(self):
            calls.append("listener")

    class Runtime:
        def on_enter_ready(self):
            calls.append("runtime")

    machine = Chart(model=Model(), listeners=[Listener()])
    machine.add_listener(Runtime())
    machine.send("go")

    assert calls == ["machine", "model", "listener", "runtime"]
