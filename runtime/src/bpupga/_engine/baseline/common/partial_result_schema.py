"""Read three-way partial-alignment outputs without relabeling legacy data.

Schema-v2 artifacts identify Native, Common and Method-aware outputs directly.
Older artifacts are handled conservatively: only fields whose recorded
provenance determines their meaning are mapped into a three-way slot.  In
particular, ``evaluation.native_pair`` is never treated as Native because old
runs used that name for the selected/final hard output.
"""

from __future__ import annotations

from typing import Any, Mapping


HARD_OUTPUT_VARIANTS = (
    "native",
    "common_decoder",
    "method_aware_decoder",
)


def _mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _missing(reason: str) -> dict[str, object]:
    return {"status": "not_applicable", "reason": reason}


def _available(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.setdefault("status", "available")
    output.setdefault("reason", None)
    return output


def _legacy_decoder_kind(decoder: Mapping[str, Any] | None) -> str | None:
    if decoder is None:
        return None
    scope = str(decoder.get("adapter_scope", "")).strip().lower()
    name = str(decoder.get("name", "")).strip().lower()
    strategy = str(decoder.get("strategy", "")).strip().lower()
    if (
        scope in {"generic_fixed_protocol", "common_cross_method_control"}
        or "fixed_mutual_top_fraction" in name
        or "fixed_mutual_top_fraction" in strategy
    ):
        return "common_decoder"
    if scope in {
        "method_specific",
        "method_native",
        "method_native_identity",
    }:
        return "method_aware_decoder"
    return None


def normalize_hard_outputs(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Native/Common/Method-aware records with explicit availability.

    The function does not manufacture metrics and does not assume that a
    legacy final output was Method-aware.  An unclassifiable legacy final is
    exposed under ``legacy_unclassified_output`` for audit only.
    """

    evaluation = _mapping(payload.get("evaluation")) or {}
    hard_outputs = _mapping(evaluation.get("hard_outputs"))
    if hard_outputs is not None:
        normalized: dict[str, dict[str, Any]] = {}
        for variant in HARD_OUTPUT_VARIANTS:
            record = _mapping(hard_outputs.get(variant))
            normalized[variant] = (
                _available(record)
                if record is not None
                else _missing("missing_schema_v2_output_record")
            )
        return normalized

    normalized = {
        "native": _missing("legacy_native_output_not_recorded"),
        "common_decoder": _missing("legacy_common_output_not_recorded"),
        "method_aware_decoder": _missing(
            "legacy_method_aware_output_not_recorded"
        ),
    }
    native = _mapping(evaluation.get("method_native_output"))
    if native is not None:
        normalized["native"] = _available(native)

    legacy_final = _mapping(evaluation.get("unified_partial_output"))
    decoder = _mapping(evaluation.get("partial_decoder"))
    if legacy_final is not None:
        kind = _legacy_decoder_kind(decoder)
        method_key = "".join(
            character
            for character in str(payload.get("method", "")).lower()
            if character.isalnum()
        )
        invalidated_method_aware = (
            kind == "method_aware_decoder"
            and method_key in {"grampa", "immnc"}
        )
        if kind is None or invalidated_method_aware:
            normalized["legacy_unclassified_output"] = {
                **_available(legacy_final),
                "reason": (
                    "legacy_method_aware_adapter_is_not_validated"
                    if invalidated_method_aware
                    else "legacy_decoder_provenance_is_ambiguous"
                ),
            }
            if invalidated_method_aware:
                normalized["method_aware_decoder"] = _missing(
                    "legacy_method_aware_adapter_is_not_validated"
                )
        else:
            normalized[kind] = _available(legacy_final)
    return normalized


def decoder_records(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return decoder provenance records for schema-v2 and legacy artifacts."""

    evaluation = _mapping(payload.get("evaluation")) or {}
    records = _mapping(evaluation.get("decoders"))
    if records is not None:
        return {
            variant: (_mapping(records.get(variant)) or {})
            for variant in ("common_decoder", "method_aware_decoder")
        }

    decoder = _mapping(evaluation.get("partial_decoder"))
    kind = _legacy_decoder_kind(decoder)
    output = {"common_decoder": {}, "method_aware_decoder": {}}
    if decoder is not None and kind is not None:
        output[kind] = decoder
    return output


__all__ = [
    "HARD_OUTPUT_VARIANTS",
    "decoder_records",
    "normalize_hard_outputs",
]
