import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

COMPONENT_NAMES = (
    "system_prompt",
    "planning_policy",
    "code_generation_policy",
    "workflow_graph",
    "verification_policy",
)


@pytest.fixture
def api():
    return importlib.import_module("tc_mo_me.genome")


def graph_data(**overrides):
    data = {
        "entry_id": "plan",
        "exit_ids": ["verify"],
        "max_nodes": 8,
        "nodes": [
            {"id": "plan", "type": "planner", "config": {}},
            {"id": "generate", "type": "generator", "config": {}},
            {"id": "verify", "type": "verifier", "config": {}},
        ],
        "edges": [
            {"source": "plan", "target": "generate"},
            {"source": "generate", "target": "verify"},
        ],
    }
    data.update(overrides)
    return data


def genome_data(api, **overrides):
    data = {
        "schema_version": "0.1",
        "system_prompt": "Solve tasks safely and verify the result.",
        "planning_policy": {"strategy": "decompose", "max_steps": 6},
        "code_generation_policy": {"language": "python", "tests_required": True},
        "workflow_graph": api.WorkflowGraph.model_validate_json(json.dumps(graph_data())),
        "verification_policy": {"require_checks": True, "stop_on_failure": False},
        "identity": {"genome_id": "test", "display_name": "Test genome"},
        "lineage": {},
    }
    data.update(overrides)
    return data


def make_genome(api, **overrides):
    return api.Genome(**genome_data(api, **overrides))


def validate_graph(api, data):
    return api.WorkflowGraph.model_validate_json(json.dumps(data))


def test_genome_serialization_round_trip(api):
    genome = make_genome(api, system_prompt="规划并验证： café 🚀")

    restored = api.Genome.model_validate_json(genome.model_dump_json())

    assert restored == genome
    assert restored.workflow_graph.entry_id == "plan"


def test_canonical_json_is_utf8_sorted_and_excludes_metadata(api):
    genome = make_genome(api, system_prompt="验证 café 🚀")

    canonical = api.canonical_genome_bytes(genome)
    decoded = canonical.decode("utf-8")
    payload = json.loads(decoded)

    assert "验证 café 🚀" in decoded
    assert "\\u" not in decoded
    assert list(payload) == sorted(payload)
    assert set(payload) == {
        "schema_version",
        "system_prompt",
        "planning_policy",
        "code_generation_policy",
        "workflow_graph",
        "verification_policy",
    }
    assert canonical == json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert api.genome_hash(genome) == hashlib.sha256(canonical).hexdigest()
    assert genome.canonical_genome_bytes() == canonical
    assert genome.genome_hash() == api.genome_hash(genome)


def test_policy_key_order_does_not_change_hash(api):
    first = make_genome(api, planning_policy={"alpha": 1, "beta": {"x": 2, "y": 3}})
    second = make_genome(api, planning_policy={"beta": {"y": 3, "x": 2}, "alpha": 1})

    assert api.canonical_genome_bytes(first) == api.canonical_genome_bytes(second)
    assert api.genome_hash(first) == api.genome_hash(second)


def test_identity_and_lineage_do_not_change_hash(api):
    original = make_genome(api)
    perturbed = make_genome(
        api,
        identity={"genome_id": "different", "arbitrary": [1, 2, 3]},
        lineage={"operator_id": "ignored", "note": "metadata only"},
    )

    assert api.genome_hash(original) == api.genome_hash(perturbed)
    assert api.canonical_genome_bytes(original) == api.canonical_genome_bytes(perturbed)


def test_content_change_changes_hash(api):
    original = make_genome(api)
    changed = make_genome(api, verification_policy={"require_checks": False})

    assert api.genome_hash(original) != api.genome_hash(changed)


def test_validate_genome_accepts_json_and_rechecks_graph(api):
    genome = make_genome(api)
    assert api.validate_genome(genome.model_dump_json()) == genome

    invalid = json.loads(genome.model_dump_json())
    invalid["workflow_graph"]["entry_id"] = "missing"
    with pytest.raises(ValueError, match="entry"):
        api.validate_genome(json.dumps(invalid))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data["nodes"].append(deepcopy(data["nodes"][0])), "Unique|unique|duplicate"),
        (lambda data: data.update(entry_id="missing"), "entry"),
        (lambda data: data.update(exit_ids=["missing"]), "exit"),
        (lambda data: data["edges"].append({"source": "missing", "target": "verify"}), "source"),
        (lambda data: data["edges"].append({"source": "plan", "target": "missing"}), "target"),
        (
            lambda data: data["nodes"].append(
                {"id": "orphan", "type": "executor", "config": {}}
            ),
            "reachable|entry",
        ),
        (
            lambda data: (
                data["nodes"].append({"id": "dead", "type": "executor", "config": {}}),
                data["edges"].append({"source": "plan", "target": "dead"}),
            ),
            "exit|reachable",
        ),
        (lambda data: data.update(max_nodes=2), "max_nodes|maximum"),
        (lambda data: data.update(max_nodes=65), "max_nodes|64|less than"),
        (lambda data: data["nodes"][0].update(type="unrestricted"), "type|planner|literal"),
        (lambda data: data.update(exit_ids=[]), "exit|at least"),
        (lambda data: data.update(nodes=[]), "nodes|at least"),
        (lambda data: data.update(exit_ids=["verify", "verify"]), "Unique|unique|duplicate"),
        (lambda data: data["edges"].append(deepcopy(data["edges"][0])), "Unique|unique|duplicate"),
    ],
)
def test_invalid_workflow_graphs_are_rejected(api, mutation, match):
    data = graph_data()
    mutation(data)

    with pytest.raises(ValueError, match=match):
        validate_graph(api, data)


def test_workflow_cycles_are_allowed_when_all_nodes_can_exit(api):
    data = graph_data()
    data["edges"].append({"source": "verify", "target": "generate"})

    graph = validate_graph(api, data)

    assert len(graph.edges) == 3


def test_schema_version_and_models_are_strict(api):
    with pytest.raises(ValueError, match="schema_version"):
        make_genome(api, schema_version="0.2")
    with pytest.raises(ValueError):
        make_genome(api, unknown_field=True)

    data = graph_data()
    data["nodes"][0]["id"] = 1
    with pytest.raises(ValueError):
        validate_graph(api, data)


def usage_data(**overrides):
    data = {
        "model_id": "test-model",
        "model_version": "test-model-2026-01",
        "price_id": "price-v1",
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "call_count": 2,
        "retry_count": 1,
        "elapsed_seconds": 0.25,
        "cost": None,
        "cost_basis": "unknown",
    }
    data.update(overrides)
    return data


def lineage_data(api, **overrides):
    parent = make_genome(api)
    child = make_genome(api, system_prompt="A mutated system prompt")
    data = {
        "operator_id": "mutation.system_prompt.v1",
        "parent_ids": ["parent"],
        "primary_parent_id": "parent",
        "child_id": "child",
        "modified_paths": ["system_prompt"],
        "component_sources": {name: "parent" for name in api.CONTENT_COMPONENTS},
        "parent_hashes": {"parent": api.genome_hash(parent)},
        "child_hash": api.genome_hash(child),
        "generation": 1,
        "usage": api.UsageRecord(**usage_data()),
        "cost": None,
        "created_at_utc": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    }
    data.update(overrides)
    return data


def test_usage_record_distinguishes_provenance_and_unknown_cost(api):
    usage = api.UsageRecord(**usage_data())

    assert usage.model_id == "test-model"
    assert usage.model_version == "test-model-2026-01"
    assert usage.price_id == "price-v1"
    assert usage.total_tokens == 18
    assert usage.call_count == 2
    assert usage.retry_count == 1
    assert usage.elapsed_seconds == 0.25
    assert usage.cost is None
    assert usage.cost_basis == "unknown"


def test_unknown_cost_cannot_be_recorded_as_zero(api):
    with pytest.raises(ValueError, match="cost|unknown"):
        api.UsageRecord(**usage_data(cost=Decimal("0"), cost_basis="unknown"))

    known_free_call = api.UsageRecord(
        **usage_data(cost=Decimal("0"), cost_basis="provider_reported")
    )
    assert known_free_call.cost == Decimal("0")


def test_usage_record_validates_token_totals(api):
    with pytest.raises(ValueError, match="total_tokens"):
        api.UsageRecord(**usage_data(total_tokens=99))


def test_lineage_records_complete_parent_child_relationship(api):
    record = api.LineageRecord(**lineage_data(api))

    assert record.operator_id == "mutation.system_prompt.v1"
    assert record.parent_ids == ["parent"]
    assert record.primary_parent_id == "parent"
    assert record.child_id == "child"
    assert record.modified_paths == ["system_prompt"]
    assert set(record.component_sources) == set(api.CONTENT_COMPONENTS)
    assert len(record.parent_hashes["parent"]) == 64
    assert len(record.child_hash) == 64
    assert record.generation == 1
    assert record.usage.cost is None
    assert record.cost is None
    assert record.created_at_utc.utcoffset() == timedelta(0)
    assert api.LineageRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"parent_ids": ["parent", "parent"]}, "unique"),
        ({"primary_parent_id": "other"}, "primary_parent_id"),
        ({"child_id": "parent"}, "child_id"),
        ({"modified_paths": []}, "modified_paths|at least"),
        ({"modified_paths": ["identity.genome_id"]}, "modified_paths|content"),
        ({"component_sources": {}}, "component_sources"),
        (
            {"component_sources": {name: "other" for name in COMPONENT_NAMES}},
            "component_sources|parent",
        ),
        ({"parent_hashes": {}}, "parent_hashes"),
        ({"parent_hashes": {"parent": "not-a-hash"}}, "parent_hashes|SHA|hash"),
        ({"child_hash": "not-a-hash"}, "child_hash|SHA|hash"),
        ({"generation": 0}, "generation"),
        ({"cost": Decimal("0")}, "cost|usage"),
        ({"created_at_utc": datetime(2026, 1, 2)}, "UTC|timezone"),
        (
            {"created_at_utc": datetime(2026, 1, 2, tzinfo=timezone(timedelta(hours=8)))},
            "UTC|timezone",
        ),
    ],
)
def test_invalid_lineage_records_are_rejected(api, overrides, match):
    data = lineage_data(api, **overrides)

    with pytest.raises(ValueError, match=match):
        api.LineageRecord(**data)


def model_request(api, **overrides):
    data = {
        "model_id": "mock-model",
        "system_prompt": "Be precise.",
        "prompt": "Return a plan",
        "temperature": 0.0,
        "max_output_tokens": 32,
    }
    data.update(overrides)
    return api.ModelRequest(**data)


def test_mock_adapter_is_deterministic_and_has_unknown_cost(api):
    adapter = api.MockModelAdapter(
        response_text="fallback", responses={"Return a plan": "1. inspect"}, model_version="mock-v1"
    )
    request = model_request(api)

    first = adapter.generate(request)
    second = adapter.generate(request)

    assert isinstance(adapter, api.ModelAdapter)
    assert first == second
    assert first.text == "1. inspect"
    assert first.model_id == "mock-model"
    assert first.model_version == "mock-v1"
    assert first.usage.input_tokens == 5
    assert first.usage.output_tokens == 2
    assert first.usage.total_tokens == 7
    assert first.usage.call_count == 1
    assert first.usage.retry_count == 0
    assert first.usage.elapsed_seconds == 0
    assert first.usage.cost is None
    assert first.usage.cost_basis == "unknown"


def test_mock_adapter_uses_deterministic_fallback(api):
    adapter = api.MockModelAdapter(response_text="fallback")

    response = adapter.generate(model_request(api, prompt="Unmapped prompt"))

    assert response.text == "fallback"


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-123",
            model="mock-model",
            system_fingerprint="fp-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="generated text"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_compatible_adapter_maps_request_response_without_network(api):
    client = FakeClient()
    adapter = api.OpenAICompatibleModelAdapter(
        client=client, model_version="fallback-version", price_id="price-v2"
    )

    response = adapter.generate(model_request(api))

    assert client.completions.kwargs == {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Return a plan"},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    assert response.text == "generated text"
    assert response.model_id == "mock-model"
    assert response.model_version == "fp-test"
    assert response.provider_response_id == "response-123"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15
    assert response.usage.price_id == "price-v2"
    assert response.usage.cost is None
    assert response.usage.cost_basis == "unknown"
    assert response.usage.call_count == 1
    assert response.usage.retry_count == 0
    assert response.usage.elapsed_seconds >= 0


def test_openai_compatible_adapter_requires_token_usage(api):
    client = FakeClient()

    class NoUsageCompletions(FakeCompletions):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.usage = None
            return response

    client.chat = SimpleNamespace(completions=NoUsageCompletions())
    adapter = api.OpenAICompatibleModelAdapter(client=client)

    with pytest.raises(ValueError, match="usage|token"):
        adapter.generate(model_request(api))


def load_initial_genomes(api):
    config_dir = Path(__file__).parents[1] / "configs" / "e1" / "genomes"
    genomes = {}
    for genome_id in ("g0_baseline", "g1_planner", "g2_verifier", "g3_workflow"):
        path = config_dir / f"{genome_id}.json"
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        assert raw == json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        genomes[genome_id] = api.validate_genome(raw)
    return genomes


def test_initial_genomes_are_canonical_non_evolved_and_unique(api):
    genomes = load_initial_genomes(api)

    assert set(genomes) == {"g0_baseline", "g1_planner", "g2_verifier", "g3_workflow"}
    assert len({api.genome_hash(genome) for genome in genomes.values()}) == 4
    for genome_id, genome in genomes.items():
        assert genome.identity["genome_id"] == genome_id
        assert genome.identity["origin"] == "manual_non_evolved"
        assert genome.lineage == {}
        assert not {"score", "fitness", "task_result"} & set(genome.identity)


def test_initial_variants_change_only_their_named_component(api):
    genomes = load_initial_genomes(api)
    baseline = genomes["g0_baseline"]
    unchanged_components = {
        "g1_planner": ("code_generation_policy", "workflow_graph", "verification_policy"),
        "g2_verifier": ("code_generation_policy", "planning_policy", "workflow_graph"),
        "g3_workflow": ("code_generation_policy", "planning_policy", "verification_policy"),
    }
    changed_component = {
        "g1_planner": "planning_policy",
        "g2_verifier": "verification_policy",
        "g3_workflow": "workflow_graph",
    }

    for genome_id in ("g1_planner", "g2_verifier", "g3_workflow"):
        variant = genomes[genome_id]
        assert variant.system_prompt == baseline.system_prompt
        for component in unchanged_components[genome_id]:
            assert getattr(variant, component) == getattr(baseline, component)
        component = changed_component[genome_id]
        assert getattr(variant, component) != getattr(baseline, component)
