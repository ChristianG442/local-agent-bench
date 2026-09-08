from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Profile:
    name: str
    host_id: str
    endpoint: str
    models: list[str]
    suites: list[str]
    contexts: list[int]
    quantizations: list[str]
    kv_cache_types: list[str]
    performance_repeats: int
    tool_repeats: int
    temperature: float
    seed: int
    gpu_required: bool
    max_ram_fraction: float
    min_ollama_version: str
    catalog: str
    hermes_max_stage: int
    hermes_min_stage_score: float
    think: bool | str | None


def load_objective_profile(path: Path) -> dict[str, Any]:
    """Load and normalise a recommendation objective profile.

    Objective profiles intentionally use a smaller schema than host profiles:
    they describe how already-collected JSONL measurements should be ranked.
    Keeping the normalisation here makes custom profiles work without changing
    the benchmark runner.
    """
    with path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    scoring = data.get("scoring", {})
    # Accept both the documented top-level sections and a nested [scoring]
    # section so hand-written profiles remain easy to evolve.
    weights = dict(data.get("weights", scoring.get("weights", {})))
    if not weights:
        weights = {
            key.removesuffix("_weight"): value
            for key, value in scoring.items()
            if key.endswith("_weight")
        }
    quality = dict(data.get("quality", scoring.get("quality", {})))
    targets = dict(data.get("targets", scoring.get("targets", {})))
    gates = dict(data.get("gates", scoring.get("gates", {})))
    return {
        "name": str(data.get("name", path.stem)),
        "weights": {str(key): float(value) for key, value in weights.items()},
        "quality_weights": {
            str(key): float(value)
            for key, value in quality.items()
        },
        "targets": {str(key): float(value) for key, value in targets.items()},
        "gates": gates,
        "required_quality_suites": [
            str(value)
            for value in data.get(
                "required_quality_suites",
                scoring.get("required_quality_suites", []),
            )
        ],
    }


def load_profile(path: Path) -> Profile:
    with path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    host = data.get("host", {})
    runtime = data.get("runtime", {})
    policy = data.get("policy", {})
    think = policy.get("think")
    if isinstance(think, str):
        normalised_think = think.strip().casefold()
        if normalised_think in {"true", "false"}:
            think = normalised_think == "true"
        elif normalised_think in {"low", "medium", "high", "max"}:
            think = normalised_think
        elif normalised_think in {"auto", "default", ""}:
            think = None
        else:
            raise ValueError(f"Unsupported think mode: {think}")
    elif think is not None and not isinstance(think, bool):
        raise ValueError(f"Unsupported think mode: {think}")
    return Profile(
        name=data["name"],
        host_id=host["id"],
        endpoint=runtime.get("endpoint", "http://localhost:11434"),
        models=list(data.get("models", [])),
        suites=list(policy.get("suites", ["smoke", "german", "tools"])),
        contexts=[int(value) for value in policy.get("contexts", [8192])],
        quantizations=list(policy.get("quantizations", ["detected"])),
        kv_cache_types=list(policy.get("kv_cache_types", ["f16"])),
        performance_repeats=int(policy.get("performance_repeats", 3)),
        tool_repeats=int(policy.get("tool_repeats", 5)),
        temperature=float(policy.get("temperature", 0)),
        seed=int(policy.get("seed", 42)),
        gpu_required=bool(host.get("gpu_required", False)),
        max_ram_fraction=float(policy.get("max_ram_fraction", 0.7)),
        min_ollama_version=str(policy.get("min_ollama_version", "0.13.0")),
        catalog=str(data.get("catalog", "models.toml")),
        hermes_max_stage=max(1, min(4, int(policy.get("hermes_max_stage", 4)))),
        hermes_min_stage_score=float(policy.get("hermes_min_stage_score", 0.70)),
        think=think,
    )


def profile_path(name_or_path: str, base_dir: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    named = base_dir / "profiles" / f"{name_or_path}.toml"
    if named.exists():
        return named
    raise FileNotFoundError(f"Profile not found: {name_or_path}")