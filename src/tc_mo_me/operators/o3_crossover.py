from __future__ import annotations

from tc_mo_me.genome import CONTENT_COMPONENTS, genome_hash, validate_genome
from tc_mo_me.operators.base import (
    EvolutionOperator,
    MutationSpec,
    OperatorContext,
    OperatorResult,
    ParentInput,
    _OperatorFailure,
    build_success,
    content_payload,
    failure_result,
    normalize_parents,
    parent_id,
    validate_context,
    zero_usage,
)


class O3CrossoverOperator(EvolutionOperator):
    operator_id = "O3"

    def generate(
        self,
        parents: ParentInput,
        spec: MutationSpec,
        context: OperatorContext,
    ) -> OperatorResult:
        usage = zero_usage()
        try:
            validate_context(context)
            normalized = normalize_parents(parents, minimum=3, maximum=None)
            ids = [parent_id(parent) for parent in normalized]
            sources = spec.component_sources
            if (
                spec.component is not None
                or spec.operation is not None
                or set(sources) != set(CONTENT_COMPONENTS)
                or any(source not in ids for source in sources.values())
            ):
                raise _OperatorFailure(
                    "invalid_spec",
                    "O3 requires exactly one valid donor id for every Genome component",
                    details={
                        "required_components": list(CONTENT_COMPONENTS),
                        "parent_ids": ids,
                    },
                )
            primary = normalized[0]
            primary_id = parent_id(primary)
            by_id = {parent_id(parent): parent for parent in normalized}
            candidate_payload = content_payload(primary)
            for component in CONTENT_COMPONENTS:
                donor = by_id[sources[component]]
                candidate_payload[component] = donor.model_dump(mode="json", include={component})[
                    component
                ]
            candidate = validate_genome(candidate_payload)
            child_hash = genome_hash(candidate)
            parent_hashes = {parent_id(parent): genome_hash(parent) for parent in normalized}
            if child_hash in set(parent_hashes.values()):
                raise _OperatorFailure(
                    "no_change",
                    "O3 recombination must produce content distinct from every parent",
                    details={"child_hash": child_hash},
                )
            modified_paths = [
                component for component in CONTENT_COMPONENTS if sources[component] != primary_id
            ]
            return build_success(
                operator_id=self.operator_id,
                parents=normalized,
                primary_parent=primary,
                candidate=candidate,
                component_sources=dict(sources),
                modified_paths=modified_paths,
                spec=spec,
                context=context,
                usage=usage,
            )
        except _OperatorFailure as error:
            return failure_result(error, usage)


O3Crossover = O3CrossoverOperator

__all__ = ["O3Crossover", "O3CrossoverOperator"]
