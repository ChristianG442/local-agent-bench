from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import HostInfo, ModelInfo


@dataclass
class Candidate:
    name: str
    display_name: str
    estimated_weight_gb: float
    min_context: int
    german: bool
    tools: bool
    roles: list[str]
    cpu_stage: int
    gpu_stage: int
    cpu_priority: int
    gpu_priority: int
    source: str
    notes: str = ""


@dataclass
class PlannedModel:
    candidate: Candidate
    installed: bool
    fit: str
    reason: str


def load_catalog(path: Path) -> list[Candidate]:
    with path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    candidates: list[Candidate] = []
    for name, raw in data.get("candidates", {}).items():
        candidates.append(
            Candidate(
                name=name,
                display_name=str(raw.get("display_name", name)),
                estimated_weight_gb=float(raw["estimated_weight_gb"]),
                min_context=int(raw.get("min_context", 8192)),
                german=bool(raw.get("german", False)),
                tools=bool(raw.get("tools", False)),
                roles=list(raw.get("roles", [])),
                cpu_stage=int(raw.get("cpu_stage", raw.get("stage", 2))),
                gpu_stage=int(raw.get("gpu_stage", raw.get("stage", 2))),
                cpu_priority=int(raw.get("cpu_priority", 100)),
                gpu_priority=int(raw.get("gpu_priority", 100)),
                source=str(raw.get("source", "")),
                notes=str(raw.get("notes", "")),
            )
        )
    return candidates


def _installed(candidate: Candidate, installed: dict[str, ModelInfo]) -> bool:
    if candidate.name in installed:
        return True
    family = candidate.name.split(":", 1)[0]
    return any(model.name.split(":", 1)[0] == family for model in installed.values())


def plan_models(host: HostInfo, candidates: list[Candidate], installed: list[ModelInfo], stage: int = 1) -> list[PlannedModel]:
    installed_by_name = {model.name: model for model in installed}
    ram_gb = host.ram_total_mib / 1024
    vram_gb = max((float(gpu["memory_total_mib"]) for gpu in host.gpu), default=0) / 1024
    is_gpu = vram_gb > 0
    # Keep headroom for KV cache, runtime overhead and the operating system.
    native_budget = vram_gb * 0.78 if is_gpu else ram_gb * 0.62
    offload_budget = vram_gb * 0.78 + ram_gb * 0.50 if is_gpu else native_budget
    eligible: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        host_stage = candidate.gpu_stage if is_gpu else candidate.cpu_stage
        if host_stage <= 0 or host_stage > stage:
            continue
        if candidate.estimated_weight_gb <= native_budget:
            fit = "full-gpu" if is_gpu else "cpu"
        elif is_gpu and candidate.estimated_weight_gb <= offload_budget:
            fit = "offload"
        else:
            continue
        eligible.append((candidate, fit))

    max_models = 5 if not is_gpu else 6
    if stage >= 2:
        max_models += 3
    priority_key = (lambda item: item[0].gpu_priority) if is_gpu else (lambda item: item[0].cpu_priority)
    selected: list[tuple[Candidate, str, str]] = []
    for candidate, fit in sorted(eligible, key=lambda item: (priority_key(item), item[0].estimated_weight_gb))[:max_models]:
        selected.append((candidate, fit, f"Rollen: {', '.join(candidate.roles)}"))

    return [
        PlannedModel(
            candidate=candidate,
            installed=_installed(candidate, installed_by_name),
            fit=fit,
            reason=reason,
        )
        for candidate, fit, reason in selected
    ]