import importlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

import tc_mo_me.model as model_module
from tc_mo_me.genome import CONTENT_COMPONENTS, Genome, WorkflowGraph, genome_hash
from tc_mo_me.model import CostBasis, MockModelAdapter, ModelAdapter

TIMESTAMP = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)


def load_base():
    return importlib.import_module("tc_mo_me.operators.base")


def load_operator(module_name):
    return importlib.import_module(f"tc_mo_me.operators.{module_name}")


def graph_data(nodes=None, edges=None):
    return {
        "entry_id": "plan",
        "exit_ids": ["verify"],
        "max_nodes": 8,
        "nodes": nodes
        or [
            {"id": "plan", "type": "planner", "config": {}},
            {"id": "generate", "type": "generator", "config": {}},
            {"id": "verify", "type": "verifier", "config": {}},
        ],
        "edges": edges
        if edges is not None
        else [
            {"source": "plan", "target": "generate"},
            {"source": "generate", "target": "verify"},
        ],
    }


def make_genome(genome_id="parent-a", graph=None, **overrides):
    data = {
        "schema_version": "0.1",
        "system_prompt": "Solve tasks safely and verify the result.",
        "planning_policy": {"strategy": "decompose", "max_steps": 6},
        "code_generation_policy": {"language": "python", "tests_required": True},
        "workflow_graph": WorkflowGraph.model_validate_json(json.dumps(graph or graph_data())),
        "verification_policy": {"require_checks": True, "stop_on_failure": False},
        "identity": {"genome_id": genome_id, "display_name": genome_id},
        "lineage": {},
    }
    data.update(overrides)
    return Genome.model_validate(data)


def content(genome):
    return genome.model_dump(mode="json", include=set(CONTENT_COMPONENTS))


class QueuedMockAdapter(MockModelAdapter):
    def __init__(self, *responses):
        super().__init__(
            price_id="mock-priced-v1",
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
        )
        self.responses = list(responses)
        self.requests = []

    def _generate_once(self, request, timeout_seconds):
        self.requests.append(request)
        self._script[request.prompt] = self.responses.pop(0)
        return super()._generate_once(request, timeout_seconds)


class FailingAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(max_retries=0, deterministic_timing=True)

    def _generate_once(self, request, timeout_seconds):
        raise RuntimeError("mock provider failed")


class RetryingFailingAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(
            max_retries=2,
            deterministic_timing=False,
            price_id="mock-priced-v1",
        )
        self.provider_attempts = 0

    def _generate_once(self, request, timeout_seconds):
        self.provider_attempts += 1
        raise TimeoutError("mock provider timed out")


def assert_failed(result, code):
    assert result.success is False
    assert result.ok is False
    assert result.status == "generation_failed"
    assert result.child is None
    assert result.lineage is None
    assert result.modified_paths == []
    assert result.modifications == []
    assert result.error is not None
    assert result.error.code == code
    assert result.error.message
    assert isinstance(result.error.details, dict)


def assert_zero_usage(usage):
    assert usage.calls == 0
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.api_cost == Decimal("0")
    assert usage.cost_basis is CostBasis.ESTIMATED
    assert usage.phase == "generation"


def test_public_operator_exports_are_available():
    api = importlib.import_module("tc_mo_me.operators")

    for name in (
        "OperatorContext",
        "MutationSpec",
        "OperatorError",
        "OperatorResult",
        "EvolutionOperator",
        "O1LocalOperator",
        "O2StructuralOperator",
        "O3CrossoverOperator",
    ):
        assert hasattr(api, name)


def test_o1_applies_only_the_selected_component_with_complete_audit_metadata():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    parent_before = parent.model_dump(mode="json")
    response = json.dumps({"component": "system_prompt", "value": "A new safe prompt."})
    adapter = QueuedMockAdapter(response)
    context = base.OperatorContext(
        model=adapter,
        model_id="mock-model",
        generation=4,
        child_id="child-o1",
        created_at_utc=TIMESTAMP,
        seed=7,
    )

    result = local.O1LocalOperator().generate(
        parent, base.MutationSpec(component="system_prompt"), context
    )

    assert result.success is True
    assert result.ok is True
    assert result.status == "succeeded"
    assert result.error is None
    assert result.child is not None
    assert result.child.system_prompt == "A new safe prompt."
    for component in CONTENT_COMPONENTS:
        if component != "system_prompt":
            assert getattr(result.child, component) == getattr(parent, component)
    assert parent.model_dump(mode="json") == parent_before
    assert result.child.identity["genome_id"] == "child-o1"
    assert result.child.identity["origin"] == "evolved"
    assert result.child.identity["generation"] == 4
    assert result.child.lineage["child_id"] == "child-o1"
    assert result.child.lineage["operator_id"] == "O1"
    assert result.modified_paths == ["system_prompt"]
    assert result.modifications == ["system_prompt"]

    lineage = result.lineage
    assert lineage is not None
    assert lineage.parent_ids == ["parent-a"]
    assert lineage.primary_parent_id == "parent-a"
    assert lineage.child_id == "child-o1"
    assert lineage.parent_hashes == {"parent-a": genome_hash(parent)}
    assert lineage.child_hash == genome_hash(result.child)
    assert lineage.child_hash != lineage.parent_hashes["parent-a"]
    assert lineage.component_sources == {component: "parent-a" for component in CONTENT_COMPONENTS}
    assert lineage.modified_paths == result.modified_paths
    assert lineage.generation == 4
    assert lineage.created_at_utc == TIMESTAMP
    assert lineage.usage == result.usage
    assert result.usage.calls == 1
    assert result.usage.total_tokens > 0
    assert result.usage.api_cost > 0
    assert result.usage.cost_basis is CostBasis.ESTIMATED
    assert result.usage.phase == "generation"
    assert len(adapter.requests) == 1
    assert adapter.requests[0].seed == 7
    assert adapter.requests[0].model_id == "mock-model"


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("system_prompt", "Replacement prompt"),
        ("planning_policy", {"strategy": "single", "max_steps": 2}),
        ("code_generation_policy", {"language": "python", "tests_required": False}),
        ("verification_policy", {"require_checks": False}),
    ],
)
def test_o1_accepts_each_allowed_content_component(component, value):
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    response = json.dumps({"component": component, "value": value})
    context = base.OperatorContext(
        model=QueuedMockAdapter(response), model_id="mock", child_id="child-o1"
    )

    result = local.O1LocalOperator().generate(
        [parent], base.MutationSpec(component=component), context
    )

    assert result.success is True
    assert getattr(result.child, component) == value
    for other in CONTENT_COMPONENTS:
        if other != component:
            assert getattr(result.child, other) == getattr(parent, other)


def test_o1_identical_mock_input_reproduces_the_complete_result():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    response = json.dumps({"component": "system_prompt", "value": "Deterministic prompt"})
    spec = base.MutationSpec(component="system_prompt")

    def run():
        context = base.OperatorContext(
            model=QueuedMockAdapter(response),
            model_id="mock",
            generation=2,
            child_id="stable-child",
            created_at_utc=TIMESTAMP,
            seed=11,
        )
        return local.O1LocalOperator().generate(parent, spec, context)

    first = run()
    second = run()

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_o1_invalid_json_returns_serializable_failure_and_preserves_usage_and_parent():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    parent_before = parent.model_dump(mode="json")
    context = base.OperatorContext(
        model=QueuedMockAdapter("```json\n{not valid json}\n```"),
        model_id="mock",
        child_id="unused-child",
    )

    result = local.O1LocalOperator().generate(
        parent, base.MutationSpec(component="system_prompt"), context
    )

    assert_failed(result, "invalid_json")
    assert result.usage.calls == 1
    assert result.usage.total_tokens > 0
    assert result.usage.api_cost > 0
    assert result.usage.cost_basis is CostBasis.ESTIMATED
    assert parent.model_dump(mode="json") == parent_before
    assert base.OperatorResult.model_validate_json(result.model_dump_json()) == result


def test_o1_rejects_workflow_or_whole_genome_rewrites():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    whole_genome = content(parent)
    whole_genome["system_prompt"] = "unauthorized rewrite"
    responses = [
        json.dumps({"component": "workflow_graph", "value": graph_data()}),
        json.dumps(whole_genome),
    ]

    for response in responses:
        context = base.OperatorContext(model=QueuedMockAdapter(response), model_id="mock")
        result = local.O1LocalOperator().generate(
            parent, base.MutationSpec(component="system_prompt"), context
        )
        assert_failed(result, "unauthorized_modification")
        assert result.usage.calls == 1
        assert result.child is None


def test_o1_rejects_component_mismatch_and_noop_without_changing_the_parent():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    before = parent.model_dump(mode="json")
    cases = [
        (
            {"component": "planning_policy", "value": {"strategy": "other"}},
            "component_mismatch",
        ),
        (
            {"component": "system_prompt", "value": parent.system_prompt},
            "no_change",
        ),
    ]

    for payload, code in cases:
        context = base.OperatorContext(
            model=QueuedMockAdapter(json.dumps(payload)), model_id="mock"
        )
        result = local.O1LocalOperator().generate(
            parent, base.MutationSpec(component="system_prompt"), context
        )
        assert_failed(result, code)

    assert parent.model_dump(mode="json") == before


def test_o1_invalid_spec_and_missing_model_fail_before_any_model_call():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    adapter = QueuedMockAdapter(json.dumps({"component": "system_prompt", "value": "new"}))

    workflow = local.O1LocalOperator().generate(
        parent,
        base.MutationSpec(component="workflow_graph"),
        base.OperatorContext(model=adapter, model_id="mock"),
    )
    missing = local.O1LocalOperator().generate(
        parent,
        base.MutationSpec(component="system_prompt"),
        base.OperatorContext(model=None, model_id="mock"),
    )

    assert_failed(workflow, "invalid_spec")
    assert_zero_usage(workflow.usage)
    assert adapter.requests == []
    assert_failed(missing, "model_unavailable")
    assert_zero_usage(missing.usage)


def test_o1_model_exception_is_a_structured_failure_with_attempted_usage():
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()

    result = local.O1LocalOperator().generate(
        parent,
        base.MutationSpec(component="system_prompt"),
        base.OperatorContext(model=FailingAdapter(), model_id="mock"),
    )

    assert_failed(result, "model_error")
    assert result.usage.calls == 1
    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.api_cost is None
    assert result.usage.cost_basis is CostBasis.UNKNOWN
    assert result.usage.phase == "generation"
    assert "mock provider failed" in result.error.message


def test_o1_retry_exhaustion_preserves_model_attempts_timing_and_usage(
    monkeypatch: pytest.MonkeyPatch,
):
    base = load_base()
    local = load_operator("o1_local")
    parent = make_genome()
    adapter = RetryingFailingAdapter()
    monkeypatch.setattr(model_module.time, "monotonic", iter([20.0, 21.5]).__next__)

    result = local.O1LocalOperator().generate(
        parent,
        base.MutationSpec(component="system_prompt"),
        base.OperatorContext(model=adapter, model_id="mock"),
    )

    assert_failed(result, "model_error")
    assert adapter.provider_attempts == 3
    assert result.usage.calls == 1
    assert result.usage.retries == 2
    assert result.usage.elapsed_seconds == 1.5
    assert result.usage.price_id == "mock-priced-v1"
    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.api_cost is None
    assert result.usage.cost_basis is CostBasis.UNKNOWN
    assert result.usage.phase == "generation"
    assert "mock provider timed out" in result.error.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", "1"),
        ("generation", 1.5),
        ("child_id", 123),
        ("created_at_utc", "2026-02-03T04:05:06+00:00"),
        ("created_at_utc", None),
    ],
)
def test_invalid_context_field_types_return_structured_failures(field, value):
    base = load_base()
    local = load_operator("o1_local")
    context = base.OperatorContext(model=None, model_id="mock", **{field: value})

    result = local.O1LocalOperator().generate(
        make_genome(),
        base.MutationSpec(component="system_prompt"),
        context,
    )

    assert_failed(result, "invalid_context")
    assert_zero_usage(result.usage)


@pytest.mark.parametrize("operation", ["insert", "replace", "reorder"])
def test_o2_applies_one_valid_graph_edit(operation):
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    before = parent.model_dump(mode="json")
    patches = {
        "insert": {
            "operation": "insert",
            "node": {"id": "execute", "type": "executor", "config": {}},
            "source": "generate",
            "target": "verify",
        },
        "replace": {
            "operation": "replace",
            "node_id": "generate",
            "node": {
                "id": "generate",
                "type": "generator",
                "config": {"style": "concise"},
            },
        },
        "reorder": {
            "operation": "reorder",
            "node_ids": ["verify", "generate", "plan"],
        },
    }
    context = base.OperatorContext(
        model=QueuedMockAdapter(json.dumps(patches[operation])),
        model_id="mock",
        child_id="child-o2",
        created_at_utc=TIMESTAMP,
    )

    result = structural.O2StructuralOperator().generate(
        parent, base.MutationSpec(operation=operation), context
    )

    assert result.success is True
    assert result.child is not None
    assert result.child.identity["genome_id"] == "child-o2"
    assert result.child.lineage["operator_id"] == "O2"
    assert result.modified_paths == ["workflow_graph"]
    assert result.lineage.modified_paths == ["workflow_graph"]
    assert result.lineage.parent_hashes == {"parent-a": genome_hash(parent)}
    assert result.lineage.child_hash == genome_hash(result.child)
    assert result.lineage.child_hash != genome_hash(parent)
    assert result.usage.calls == 1
    assert result.usage.phase == "generation"
    assert parent.model_dump(mode="json") == before
    for component in CONTENT_COMPONENTS:
        if component != "workflow_graph":
            assert getattr(result.child, component) == getattr(parent, component)
    if operation == "insert":
        assert [node.id for node in result.child.workflow_graph.nodes] == [
            "plan",
            "generate",
            "execute",
            "verify",
        ]
        assert [(edge.source, edge.target) for edge in result.child.workflow_graph.edges] == [
            ("plan", "generate"),
            ("generate", "execute"),
            ("execute", "verify"),
        ]
    elif operation == "replace":
        assert result.child.workflow_graph.nodes[1].config == {"style": "concise"}
    else:
        assert [node.id for node in result.child.workflow_graph.nodes] == [
            "verify",
            "generate",
            "plan",
        ]


def test_o2_delete_reconnects_a_single_intermediate_node():
    base = load_base()
    structural = load_operator("o2_structural")
    graph = graph_data(
        nodes=[
            {"id": "plan", "type": "planner", "config": {}},
            {"id": "generate", "type": "generator", "config": {}},
            {"id": "execute", "type": "executor", "config": {}},
            {"id": "verify", "type": "verifier", "config": {}},
        ],
        edges=[
            {"source": "plan", "target": "generate"},
            {"source": "generate", "target": "execute"},
            {"source": "execute", "target": "verify"},
        ],
    )
    parent = make_genome(graph=graph)
    before = parent.model_dump(mode="json")
    patch = {"operation": "delete", "node_id": "execute"}
    context = base.OperatorContext(
        model=QueuedMockAdapter(json.dumps(patch)), model_id="mock", child_id="child-o2"
    )

    result = structural.O2StructuralOperator().generate(
        parent, base.MutationSpec(operation="delete"), context
    )

    assert result.success is True
    assert [node.id for node in result.child.workflow_graph.nodes] == [
        "plan",
        "generate",
        "verify",
    ]
    assert [(edge.source, edge.target) for edge in result.child.workflow_graph.edges] == [
        ("plan", "generate"),
        ("generate", "verify"),
    ]
    assert parent.model_dump(mode="json") == before


def test_o2_identical_mock_input_reproduces_the_child_and_lineage():
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    patch = {
        "operation": "replace",
        "node_id": "generate",
        "node": {"id": "generate", "type": "executor", "config": {}},
    }
    spec = base.MutationSpec(operation="replace")

    def run():
        return structural.O2StructuralOperator().generate(
            parent,
            spec,
            base.OperatorContext(
                model=QueuedMockAdapter(json.dumps(patch)),
                model_id="mock",
                child_id="stable-o2",
                created_at_utc=TIMESTAMP,
            ),
        )

    assert run() == run()


def test_o2_invalid_json_and_multiple_operations_fail_with_usage():
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    multiple = [
        {"operation": "delete", "node_id": "generate"},
        {
            "operation": "replace",
            "node_id": "plan",
            "node": {"id": "plan", "type": "planner", "config": {"new": True}},
        },
    ]

    for response, code in [
        ("not-json", "invalid_json"),
        (json.dumps(multiple), "multiple_operations"),
    ]:
        result = structural.O2StructuralOperator().generate(
            parent,
            base.MutationSpec(operation="delete"),
            base.OperatorContext(model=QueuedMockAdapter(response), model_id="mock"),
        )
        assert_failed(result, code)
        assert result.usage.calls == 1
        assert result.usage.total_tokens > 0


def test_o2_rejects_unauthorized_paths_and_operation_mismatch():
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    before = parent.model_dump(mode="json")
    cases = [
        (
            base.MutationSpec(operation="replace"),
            {
                "operation": "replace",
                "path": "/system_prompt",
                "value": "unauthorized",
            },
            "unauthorized_modification",
        ),
        (
            base.MutationSpec(operation="replace"),
            {"operation": "delete", "node_id": "generate"},
            "operation_mismatch",
        ),
    ]

    for spec, patch, code in cases:
        result = structural.O2StructuralOperator().generate(
            parent,
            spec,
            base.OperatorContext(model=QueuedMockAdapter(json.dumps(patch)), model_id="mock"),
        )
        assert_failed(result, code)

    assert parent.model_dump(mode="json") == before


def test_o2_invalid_graph_is_rejected_by_genome_validation():
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    before = parent.model_dump(mode="json")
    duplicate_insert = {
        "operation": "insert",
        "node": {"id": "generate", "type": "executor", "config": {}},
        "source": "generate",
        "target": "verify",
    }

    result = structural.O2StructuralOperator().generate(
        parent,
        base.MutationSpec(operation="insert"),
        base.OperatorContext(
            model=QueuedMockAdapter(json.dumps(duplicate_insert)), model_id="mock"
        ),
    )

    assert_failed(result, "invalid_genome")
    assert result.usage.calls == 1
    assert parent.model_dump(mode="json") == before


def test_o2_noop_and_invalid_spec_do_not_produce_children():
    base = load_base()
    structural = load_operator("o2_structural")
    parent = make_genome()
    original_node = parent.workflow_graph.nodes[0].model_dump(mode="json")

    noop = structural.O2StructuralOperator().generate(
        parent,
        base.MutationSpec(operation="replace"),
        base.OperatorContext(
            model=QueuedMockAdapter(
                json.dumps(
                    {
                        "operation": "replace",
                        "node_id": "plan",
                        "node": original_node,
                    }
                )
            ),
            model_id="mock",
        ),
    )
    invalid_spec = structural.O2StructuralOperator().generate(
        parent,
        base.MutationSpec(operation="replace", component="system_prompt"),
        base.OperatorContext(model=None, model_id="mock"),
    )

    assert_failed(noop, "no_change")
    assert_failed(invalid_spec, "invalid_spec")
    assert_zero_usage(invalid_spec.usage)


def crossover_parents():
    primary = make_genome("p1")
    reordered_graph = graph_data(
        nodes=[
            {"id": "verify", "type": "verifier", "config": {}},
            {"id": "generate", "type": "generator", "config": {}},
            {"id": "plan", "type": "planner", "config": {}},
        ]
    )
    second = make_genome(
        "p2",
        graph=reordered_graph,
        system_prompt="Prompt inherited from p2",
        planning_policy={"strategy": "p2-only"},
    )
    third = make_genome(
        "p3",
        code_generation_policy={
            "language": "python",
            "tests_required": False,
            "style": "typed",
        },
        verification_policy={"require_checks": False, "stop_on_failure": True},
    )
    return primary, second, third


def crossover_sources():
    return {
        "system_prompt": "p2",
        "planning_policy": "p1",
        "code_generation_policy": "p3",
        "workflow_graph": "p2",
        "verification_policy": "p1",
    }


def test_o3_deterministically_recombines_whole_components_without_model_calls():
    base = load_base()
    crossover = load_operator("o3_crossover")
    parents = crossover_parents()
    before = [parent.model_dump(mode="json") for parent in parents]
    sources = crossover_sources()
    context = base.OperatorContext(
        model=None,
        generation=3,
        child_id="child-o3",
        created_at_utc=TIMESTAMP,
    )

    result = crossover.O3CrossoverOperator().generate(
        list(parents), base.MutationSpec(component_sources=sources), context
    )

    assert result.success is True
    assert result.error is None
    assert result.child is not None
    assert result.child.system_prompt == parents[1].system_prompt
    assert result.child.planning_policy == parents[0].planning_policy
    assert result.child.code_generation_policy == parents[2].code_generation_policy
    assert result.child.workflow_graph == parents[1].workflow_graph
    assert result.child.verification_policy == parents[0].verification_policy
    assert result.child.workflow_graph != parents[0].workflow_graph
    assert result.modified_paths == [
        "system_prompt",
        "code_generation_policy",
        "workflow_graph",
    ]
    assert result.child.identity["genome_id"] == "child-o3"
    assert result.child.identity["origin"] == "evolved"
    assert result.child.lineage["primary_parent_id"] == "p1"

    lineage = result.lineage
    assert lineage is not None
    assert lineage.operator_id == "O3"
    assert lineage.parent_ids == ["p1", "p2", "p3"]
    assert lineage.primary_parent_id == "p1"
    assert lineage.component_sources == sources
    assert lineage.parent_hashes == {
        parent.identity["genome_id"]: genome_hash(parent) for parent in parents
    }
    assert lineage.child_hash == genome_hash(result.child)
    assert lineage.child_hash not in set(lineage.parent_hashes.values())
    assert lineage.usage == result.usage
    assert_zero_usage(result.usage)
    assert [parent.model_dump(mode="json") for parent in parents] == before


def test_o3_modified_paths_use_content_differences_and_preserve_donor_sources():
    base = load_base()
    crossover = load_operator("o3_crossover")
    primary, second, third = crossover_parents()
    second = second.model_copy(update={"system_prompt": primary.system_prompt})
    sources = crossover_sources()

    result = crossover.O3CrossoverOperator().generate(
        [primary, second, third],
        base.MutationSpec(component_sources=sources),
        base.OperatorContext(model=None),
    )

    assert result.success is True
    assert result.child is not None
    assert result.child.system_prompt == primary.system_prompt == second.system_prompt
    assert result.modified_paths == ["code_generation_policy", "workflow_graph"]
    assert "system_prompt" not in result.modified_paths
    assert result.lineage is not None
    assert result.lineage.component_sources == sources
    assert result.lineage.component_sources["system_prompt"] == "p2"
    assert result.lineage.modified_paths == result.modified_paths


def test_o3_same_inputs_reproduce_the_complete_result():
    base = load_base()
    crossover = load_operator("o3_crossover")
    parents = crossover_parents()
    spec = base.MutationSpec(component_sources=crossover_sources())

    def run():
        return crossover.O3CrossoverOperator().generate(
            list(parents),
            spec,
            base.OperatorContext(
                generation=3,
                child_id="stable-o3",
                created_at_utc=TIMESTAMP,
            ),
        )

    first = run()
    second = run()

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_o3_requires_three_distinct_parents_and_explicit_complete_sources():
    base = load_base()
    crossover = load_operator("o3_crossover")
    parents = crossover_parents()
    cases = [
        (parents[:2], base.MutationSpec(component_sources=crossover_sources()), "invalid_parents"),
        (
            parents,
            base.MutationSpec(
                component_sources={**crossover_sources(), "system_prompt": "unknown"}
            ),
            "invalid_spec",
        ),
        (
            parents,
            base.MutationSpec(
                component_sources={
                    key: value
                    for key, value in crossover_sources().items()
                    if key != "workflow_graph"
                }
            ),
            "invalid_spec",
        ),
        (
            [parents[0], parents[0], parents[2]],
            base.MutationSpec(component_sources=crossover_sources()),
            "invalid_parents",
        ),
        (
            parents,
            base.MutationSpec(component_sources=crossover_sources(), component="system_prompt"),
            "invalid_spec",
        ),
    ]

    for selected_parents, spec, code in cases:
        result = crossover.O3CrossoverOperator().generate(
            list(selected_parents), spec, base.OperatorContext(model=None)
        )
        assert_failed(result, code)
        assert_zero_usage(result.usage)


def test_o3_rejects_a_crossover_that_only_copies_the_primary_parent():
    base = load_base()
    crossover = load_operator("o3_crossover")
    parents = crossover_parents()
    before = [parent.model_dump(mode="json") for parent in parents]
    sources = {component: "p1" for component in CONTENT_COMPONENTS}

    result = crossover.O3CrossoverOperator().generate(
        list(parents),
        base.MutationSpec(component_sources=sources),
        base.OperatorContext(model=None),
    )

    assert_failed(result, "no_change")
    assert_zero_usage(result.usage)
    assert [parent.model_dump(mode="json") for parent in parents] == before
