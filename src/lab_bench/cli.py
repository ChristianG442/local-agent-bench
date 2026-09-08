from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import time
from pathlib import Path

from .config import load_objective_profile, load_profile, profile_path
from .core import (
    OllamaAdapter,
    SUPPORTED_KV_CACHE_TYPES,
    cases_for_suites,
    detect_host,
    hermes_stage_allowed,
    local_ollama_version,
    normalise_kv_cache_type,
    PREFLIGHT_TIMEOUT_SECONDS,
    quantization_matches,
    read_jsonl,
    run_case,
    sanitize_records,
    summarize_results,
    summarize_paths,
    unsupported_kv_cache_types,
    update_instructions,
    version_at_least,
    write_jsonl,
)
from .planner import load_catalog, plan_models


ROOT = Path.cwd()
DEFAULT_RESULTS = ROOT / "results"


def _csv(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _think(value: str) -> bool | str | None:
    normalised = value.strip().casefold()
    if normalised in {"auto", "default"}:
        return "auto"
    if normalised in {"true", "false"}:
        return normalised == "true"
    if normalised in {"low", "medium", "high", "max"}:
        return normalised
    raise argparse.ArgumentTypeError(
        "think muss auto, true, false, low, medium, high oder max sein"
    )


def _preflight_error_record(
    host,
    model,
    kv_cache_type: str,
    error_type: str,
    reason: str,
    message: str,
) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host.__dict__,
        "model": model.__dict__,
        "config": {"kv_cache_type": kv_cache_type},
        "case": {"suite": "preflight", "id": "kv_cache_available"},
        "evaluation": {"passed": False, "score": 0, "reason": reason},
        "error": {"type": error_type, "message": message},
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="gpu-12gb", help="Profile name or path to a .toml profile")
    parser.add_argument("--profiles-dir", type=Path, default=ROOT, help="Directory containing profiles/")


def _load(args: argparse.Namespace):
    profile = load_profile(profile_path(args.profile, args.profiles_dir))
    endpoint = os.environ.get("OLLAMA_HOST", profile.endpoint)
    host = detect_host(endpoint, profile.host_id)
    adapter = OllamaAdapter(endpoint)
    return profile, host, adapter


def _catalog_path(profile, profiles_dir: Path) -> Path:
    candidate = Path(profile.catalog)
    if candidate.exists():
        return candidate
    candidate = profiles_dir / profile.catalog
    if candidate.exists():
        return candidate
    return profiles_dir / "models.toml"


def cmd_detect(args: argparse.Namespace) -> int:
    profile, host, adapter = _load(args)
    installed = adapter.list_models()
    print(json.dumps({"profile": profile.name, "host": host.__dict__, "ollama_cli_version": local_ollama_version(), "installed_models": [model.__dict__ for model in installed]}, ensure_ascii=False, indent=2))
    return 0 if host.runtime_version else 2


def cmd_doctor(args: argparse.Namespace) -> int:
    profile, host, adapter = _load(args)
    cli_version = local_ollama_version()
    version = cli_version or host.runtime_version
    version_ok = version_at_least(version, args.min_version or profile.min_ollama_version)
    installed = adapter.list_models()
    print(f"Profil: {profile.name}")
    print(f"Endpoint: {host.runtime_endpoint}")
    print(f"Ollama API: {host.runtime_version or 'nicht erreichbar'}")
    print(f"Ollama CLI: {cli_version or 'nicht lokal vorhanden (bei Docker normal)'}")
    print(f"Host: {host.os} | CPU: {host.cpu_threads} Threads | RAM: {host.ram_total_mib} MiB | GPUs: {len(host.gpu)}")
    print(f"Installierte Modelle: {len(installed)}")
    if not host.runtime_version:
        print("\nFEHLER: Ollama ist nicht erreichbar.")
        print("Starte Ollama oder setze OLLAMA_HOST auf den erreichbaren API-Endpunkt.")
        return 2
    if version_ok is False:
        print(f"\nFEHLER: Ollama {version} ist älter als benötigt ({args.min_version or profile.min_ollama_version}).")
        print("Sicheres Update für eine direkte Host-Installation:")
        for instruction in update_instructions():
            print(f"  {instruction}")
        print("Bei Docker stattdessen:")
        for instruction in update_instructions(docker=True):
            print(f"  {instruction}")
        return 2
    if version_ok is None:
        print("\nWARNUNG: Ollama-Version konnte nicht zuverlässig verglichen werden.")
    if profile.gpu_required and not host.gpu:
        print("\nFEHLER: Dieses Profil benötigt eine NVIDIA-GPU, aber nvidia-smi meldet keine.")
        return 2
    print("\nPreflight OK. Es wird kein Modell automatisch heruntergeladen oder aktualisiert.")
    return 0


def _print_plan(profile, host, adapter, profiles_dir: Path, stage: int) -> tuple[list[str], list[str]]:
    candidates = load_catalog(_catalog_path(profile, profiles_dir))
    planned = plan_models(host, candidates, adapter.list_models(), stage=stage)
    installed = [item for item in planned if item.installed]
    missing = [item for item in planned if not item.installed]
    print(f"Intelligenter Modellplan für {profile.name} | Stage {stage}")
    print(f"Budgetbasis: {host.ram_total_mib} MiB RAM, {len(host.gpu)} NVIDIA-GPU(s)")
    print("\nBehalten / bereits vorhanden:")
    for item in installed:
        print(f"  ✓ {item.candidate.name:<38} {item.fit:<10} — {item.reason}")
    print("\nGezielt nachladen:")
    for item in missing:
        print(f"  + {item.candidate.name:<38} {item.fit:<10} — {item.reason}")
    print("\nZurückgestellt:")
    for candidate in candidates:
        host_stage = candidate.gpu_stage if host.gpu else candidate.cpu_stage
        if host_stage > stage:
            print(f"  · {candidate.name:<38} Stage {host_stage} — {candidate.notes or 'Qualitäts-Challenger'}")
    return [item.candidate.name for item in installed], [item.candidate.name for item in missing]


def cmd_plan(args: argparse.Namespace) -> int:
    profile, host, adapter = _load(args)
    if not host.runtime_version:
        print("Ollama ist nicht erreichbar. Der Plan benötigt den lokalen Modellbestand.", file=sys.stderr)
        return 2
    if profile.gpu_required and not host.gpu:
        print("Profil benötigt eine NVIDIA-GPU, aber nvidia-smi meldet keine GPU.", file=sys.stderr)
        return 2
    _print_plan(profile, host, adapter, args.profiles_dir, args.stage)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    profile, host, adapter = _load(args)
    if not host.runtime_version:
        print("Ollama ist nicht erreichbar. Zuerst `lab doctor --profile ...` ausführen.", file=sys.stderr)
        return 2
    if profile.gpu_required and not host.gpu:
        print("Profil benötigt eine NVIDIA-GPU, aber nvidia-smi meldet keine GPU.", file=sys.stderr)
        return 2
    _, missing = _print_plan(profile, host, adapter, args.profiles_dir, args.stage)
    if not missing:
        print("\nAlle empfohlenen Modelle sind bereits vorhanden.")
        return 0
    if not args.yes:
        print("\nEs wurde nichts heruntergeladen. Für den bewussten Download erneut mit `--yes` starten.")
        return 3
    for model in missing:
        print(f"\nPull: {model}")
        ok, status = adapter.pull_model(model)
        print(f"  {'OK' if ok else 'FEHLER'} — {status}")
        if not ok:
            return 2
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    profile, host, adapter = _load(args)
    if not host.runtime_version:
        print("Ollama ist nicht erreichbar. Zuerst `lab doctor --profile ...` ausführen.", file=sys.stderr)
        return 2
    if profile.gpu_required and not host.gpu:
        print("Profil benötigt eine NVIDIA-GPU, aber nvidia-smi meldet keine GPU.", file=sys.stderr)
        return 2
    installed_list = adapter.list_models()
    installed = {model.name: model for model in installed_list}
    if args.models:
        models = _csv(args.models) or []
    else:
        planned, _ = _print_plan(profile, host, adapter, args.profiles_dir, args.stage)
        models = planned
        if not models:
            print("Keine geplanten Modelle sind installiert. Erst `lab prepare --profile ... --yes` ausführen.", file=sys.stderr)
            return 3
    suites = _csv(args.suites) or profile.suites
    contexts = [int(value) for value in args.contexts.split(",")] if args.contexts else profile.contexts
    quantizations = _csv(args.quantizations) or profile.quantizations
    kv_cache_types = _csv(args.kv_cache_types) or profile.kv_cache_types
    results_path = Path(args.output) if args.output else DEFAULT_RESULTS / f"{profile.host_id}.jsonl"
    think = profile.think if args.think is None else (
        None if args.think == "auto" else args.think
    )
    records: list[dict] = []
    print(f"Profil: {profile.name} | Host: {host.host_id} | Runtime: {host.runtime_version or 'nicht erreichbar'}")
    print(f"Modelle: {', '.join(models)}")
    print(f"Suiten: {', '.join(suites)} | Kontexte: {', '.join(map(str, contexts))}")
    print(f"Quantisierungen: {', '.join(quantizations) or 'beliebig (von Ollama erkannt)'}")
    print(f"KV-Cache: {', '.join(kv_cache_types)}")
    print(f"Thinking: {think if think is not None else 'auto (Ollama-Modellstandard)'}")
    for model_name in models:
        model = installed.get(model_name) or next((item for name, item in installed.items() if name.split(":")[0] == model_name.split(":")[0]), None)
        if not model:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": host.__dict__,
                "model": {"name": model_name, "available": False},
                "config": {"kv_cache_types": kv_cache_types},
                "case": {"suite": "preflight", "id": "model_available"},
                "evaluation": {"passed": False, "score": 0, "reason": "model_not_installed"},
                "error": {"type": "ModelNotInstalled", "message": f"{model_name} is not installed on this Ollama host"},
            }
            records.append(record)
            print(f"  {model_name}: nicht installiert (wird nicht stillschweigend heruntergeladen)")
            continue
        if not quantization_matches(model, quantizations):
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": host.__dict__,
                "model": model.__dict__,
                "config": {
                    "quantizations": quantizations,
                    "kv_cache_types": kv_cache_types,
                },
                "case": {"suite": "preflight", "id": "quantization_available"},
                "evaluation": {"passed": False, "score": 0, "reason": "quantization_mismatch"},
                "error": {
                    "type": "QuantizationMismatch",
                    "message": (
                        f"{model.name} reports {model.quantization or 'unknown'}, "
                        f"requested {', '.join(quantizations)}"
                    ),
                },
            }
            records.append(record)
            print(
                f"  {model.name}: Quantisierung {model.quantization or 'unbekannt'} "
                f"(nicht in {', '.join(quantizations)})"
            )
            continue
        print(f"  {model.name}: {model.quantization or 'Quantisierung unbekannt'}")
        for requested_kv_cache_type in kv_cache_types:
            kv_cache_type = normalise_kv_cache_type(requested_kv_cache_type)
            invalid_types = unsupported_kv_cache_types([requested_kv_cache_type])
            if invalid_types:
                message = (
                    f"{requested_kv_cache_type} ist nicht unterstützt; "
                    f"erlaubt: {', '.join(SUPPORTED_KV_CACHE_TYPES)}"
                )
                record = _preflight_error_record(
                    host,
                    model,
                    kv_cache_type,
                    "KVCacheTypeUnsupported",
                    "kv_cache_unsupported",
                    message,
                )
                records.append(record)
                print(f"    KV-Cache {requested_kv_cache_type}: PRECHECK FEHLER — {message}")
                continue
            preflight = adapter.preflight_kv_cache_type(
                model.name,
                kv_cache_type,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
            )
            if not preflight.supported:
                record = _preflight_error_record(
                    host,
                    model,
                    kv_cache_type,
                    {
                        "kv_cache_unsupported": "KVCacheRuntimeUnsupported",
                        "preflight_timeout": "PreflightTimeout",
                    }.get(preflight.status, "PreflightError"),
                    preflight.status,
                    preflight.message,
                )
                records.append(record)
                print(
                    f"    KV-Cache {kv_cache_type}: PRECHECK FEHLER "
                    f"[{preflight.status}] — {preflight.message}"
                )
                continue
            print(f"    KV-Cache {kv_cache_type}: Preflight OK")
            for context in contexts:
                hermes_max_stage = args.hermes_max_stage or profile.hermes_max_stage
                cases = cases_for_suites(
                    suites,
                    context,
                    hermes_stage=hermes_max_stage,
                )
                if args.max_cases:
                    cases = cases[: args.max_cases]
                if args.max_output_tokens:
                    cases = [
                        replace(case, max_output_tokens=args.max_output_tokens)
                        for case in cases
                    ]
                track_records: list[dict] = []
                blocked_hermes_stage: int | None = None
                for case in cases:
                    hermes_stage = int(case.metadata.get("stage", 1))
                    if case.suite == "hermes_agent" and hermes_stage > 1:
                        allowed, reason = hermes_stage_allowed(
                            hermes_stage,
                            track_records,
                            minimum_score=profile.hermes_min_stage_score,
                        )
                        if not allowed:
                            if blocked_hermes_stage != hermes_stage:
                                print(
                                    f"    Hermes Stage {hermes_stage} übersprungen — {reason}"
                                )
                            blocked_hermes_stage = hermes_stage
                            continue
                    repeats = (
                        profile.tool_repeats
                        if case.suite in {"tools", "hermes_agent"}
                        else profile.performance_repeats
                    )
                    for repeat in range(repeats):
                        record = run_case(
                            adapter,
                            host,
                            model,
                            case,
                            context_tokens=context,
                            temperature=profile.temperature,
                            seed=profile.seed,
                            repeat=repeat,
                            kv_cache_type=kv_cache_type,
                            think=think,
                        )
                        records.append(record)
                        track_records.append(record)
                        status = "OK" if record["evaluation"]["passed"] else "FAIL"
                        print(
                            f"    {case.id} [{context}] KV {kv_cache_type} "
                            f"#{repeat + 1}: {status} — {record['evaluation']['reason']}"
                        )
    count = write_jsonl(results_path, records)
    print(f"{count} Läufe gespeichert: {results_path}")
    summaries = summarize_results(
        results_path,
        load_objective_profile(profile_path("quality-first", args.profiles_dir)),
    )
    if summaries:
        recommended = next((item for item in summaries if item["eligible"]), summaries[0])
        print(
            f"\nAktuelle Empfehlung für {profile.name}: "
            f"{recommended['model']} ({recommended['total_score']:.0%} Zielscore, "
            f"quality-first)"
        )
        print("Details: lab recommend --input " + str(results_path))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    raw_inputs = args.input if isinstance(args.input, (list, tuple)) else [args.input]
    input_paths = [Path(path) for path in raw_inputs]
    objective = load_objective_profile(
        profile_path(args.objective_profile, args.profiles_dir)
    )
    summaries = summarize_paths(
        input_paths,
        objective,
        re_evaluate=args.re_evaluate,
    )
    if not summaries:
        print(f"Keine Ergebnisse gefunden: {', '.join(map(str, input_paths))}", file=sys.stderr)
        return 2
    host_ids = {item["host_id"] for item in summaries}
    if len(host_ids) > 1:
        print(
            "Ergebnisdateien enthalten mehrere Hosts "
            f"({', '.join(sorted(host_ids))}). "
            "Bitte jeden Host separat empfehlen, damit Messwerte nicht "
            "irreführend verglichen werden.",
            file=sys.stderr,
        )
        return 2
    print(f"Modellvergleich (Zielprofil: {objective['name']})")
    if args.re_evaluate:
        print("Bewertung: Rohdaten unverändert mit Benchmark 0.3.1 neu ausgewertet")
    for index, summary in enumerate(summaries, 1):
        speed = f"{summary['generation_tokens_per_second']:.1f} tok/s" if summary["generation_tokens_per_second"] else "—"
        wall = f"{summary['wall_seconds']:.2f} s Wall Time" if summary["wall_seconds"] else "—"
        memory = []
        if summary["model_size_mib"] is not None:
            memory.append(f"Gewichte {summary['model_size_mib']:.0f} MiB")
        if summary["ram_peak_mib"] is not None:
            memory.append(f"RAM-Peak {summary['ram_peak_mib']:.0f} MiB")
        if summary["vram_peak_mib"] is not None:
            memory.append(f"VRAM-Peak {summary['vram_peak_mib']:.0f} MiB")
        memory_text = ", ".join(memory) if memory else "—"
        print(
            f"{index:>2}. {'*' if summary['pareto_optimal'] else ' '} "
            f"{summary['model']:<28} "
            f"[Gewichte {summary['quantization']} | KV-Cache {summary['kv_cache_type']}]"
            f" [Thinking {summary['think']}]"
        )
        print(
            f"    Zielscore: {summary['total_score']:.0%} | "
            f"Qualität: {summary['quality_score']:.0%} | "
            f"Deutsch {summary['german_score'] if summary['german_score'] is not None else 0:.0%} | "
            f"Tools {summary['tools_score'] if summary['tools_score'] is not None else 0:.0%} | "
            f"Hermes {summary['hermes_agent_score'] if summary['hermes_agent_score'] is not None else 0:.0%} | "
            f"Long Context {summary['long_context_score'] if summary['long_context_score'] is not None else 0:.0%}"
        )
        print(
            f"    Hermes-kritische Fehler: {summary['hermes_critical_failures']}"
        )
        print(f"    Speicher: {memory_text}")
        print(
            "    Laufzustand: "
            f"kalt {summary['cold_runs']} Runs"
            f" (TTFT {summary['cold_ttft_seconds']:.2f} s)"
            if summary["cold_ttft_seconds"] is not None
            else f"    Laufzustand: kalt {summary['cold_runs']} Runs"
        )
        print(
            "    Laufzustand: "
            f"warm {summary['warm_runs']} Runs"
            f" (TTFT {summary['warm_ttft_seconds']:.2f} s)"
            if summary["warm_ttft_seconds"] is not None
            else f"    Laufzustand: warm {summary['warm_runs']} Runs"
        )
        efficiency = summary["efficiency"]
        efficiency_text = ", ".join(
            f"{label} {efficiency[key]:.0%}"
            for label, key in (
                ("Tok/s", "generation_tokens_per_second"),
                ("TTFT", "ttft_seconds"),
                ("Wall", "wall_seconds"),
                ("RAM", "ram_peak_mib"),
                ("VRAM", "vram_peak_mib"),
                ("Gewichte", "model_size_mib"),
            )
            if efficiency.get(key) is not None
        ) or "keine Zielmessung"
        print(f"    Tempo: {speed}, {wall}")
        print(f"    Effizienz (Ziel=100%): {efficiency_text}")
        if summary["gate_failures"]:
            print(
                "    Hard Gates: AUSGESCHLOSSEN — "
                + ", ".join(summary["gate_failures"])
            )
        else:
            print("    Hard Gates: OK")
    pareto = [item["model"] for item in summaries if item["pareto_optimal"]]
    print(f"\nPareto-Front (*): {', '.join(pareto) if pareto else 'keine zulässige Kombination'}")
    best = next((item for item in summaries if item["eligible"]), None)
    if best:
        print(
            f"Aktuelle Empfehlung: {best['model']} "
            f"(Zielscore {best['total_score']:.0%}, KV-Cache {best['kv_cache_type']})"
        )
    else:
        print("Aktuelle Empfehlung: keine — alle Tracks scheitern an einem Hard Gate.")
    print(
        "Effizienzwerte sind gegen die Profilziele normiert; fehlende Messwerte "
        "werden nicht stillschweigend als gut bewertet. * markiert nicht "
        "dominierte zulässige Tracks."
    )
    return 0


def cmd_sanitize(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.force:
        print(
            f"Zieldatei existiert bereits: {args.output}. "
            "Zum bewussten Überschreiben --force verwenden.",
            file=sys.stderr,
        )
        return 3
    records = read_jsonl(args.input)
    if not records:
        print(f"Keine JSONL-Records gefunden: {args.input}", file=sys.stderr)
        return 2
    sanitized = sanitize_records(
        records,
        args.host_id,
        keep_timestamps=args.keep_timestamps,
        keep_kernel=args.keep_kernel,
    )
    if args.output.exists():
        args.output.unlink()
    count = write_jsonl(args.output, sanitized)
    print(f"{count} anonymisierte Records gespeichert: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description="Portable local-agent benchmark for Ollama")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Host, Ollama und installierte Modelle erkennen")
    _common(detect)
    detect.set_defaults(func=cmd_detect)

    doctor = subparsers.add_parser("doctor", help="Ollama, Hardware und Profil vor dem Benchmark prüfen")
    _common(doctor)
    doctor.add_argument("--min-version", help="Mindestversion überschreiben (z.B. 0.13.0)")
    doctor.set_defaults(func=cmd_doctor)

    plan = subparsers.add_parser("plan", help="Kleine, hardwaregerechte Modell-Auswahl planen")
    _common(plan)
    plan.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1 = Kernset, 2 = Qualitäts-Challenger")
    plan.set_defaults(func=cmd_plan)

    prepare = subparsers.add_parser("prepare", help="Geplante Modelle über Ollama gezielt nachladen")
    _common(prepare)
    prepare.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1 = Kernset, 2 = Qualitäts-Challenger")
    prepare.add_argument("--yes", action="store_true", help="Download wirklich ausführen")
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="Benchmark-Matrix ausführen")
    _common(run)
    run.add_argument("--models", help="Kommagetrennte Ollama-Modellnamen; überschreibt das Profil")
    run.add_argument("--suites", help="Kommagetrennte Suiten; z.B. smoke,german,tools")
    run.add_argument("--contexts", help="Kommagetrennte Kontextgrößen; z.B. 8192,32768,65536")
    run.add_argument(
        "--quantizations",
        help="Kommagetrennte Quantisierungen; z.B. Q4_K_M,Q5_K_M,Q6_K,Q8_0 (detected lässt Ollama entscheiden)",
    )
    run.add_argument(
        "--kv-cache-types",
        help="Kommagetrennte KV-Cache-Typen; z.B. f16,q8_0,q4_0",
    )
    run.add_argument("--output", help="JSONL-Zieldatei")
    run.add_argument(
        "--think",
        type=_think,
        help="Ollama-Thinking explizit steuern: auto, true, false oder low/medium/high/max",
    )
    run.add_argument(
        "--max-output-tokens",
        type=int,
        help="Diagnose-Override für das gemeinsame Thinking-/Antwort-Tokenbudget",
    )
    run.add_argument("--max-cases", type=int, help="Nur die ersten N Fälle je Suite (Smoke-Test)")
    run.add_argument(
        "--hermes-max-stage",
        type=int,
        choices=[1, 2, 3, 4],
        help="Höchste Hermes-Stufe; teure Stufen werden bei schwacher Basis adaptiv ausgelassen",
    )
    run.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Automatische Auswahl: 1 = Kernset, 2 = Qualitäts-Challenger")
    run.set_defaults(func=cmd_run)

    recommend = subparsers.add_parser("recommend", help="Ergebnisse zusammenfassen und Optimum vorschlagen")
    recommend.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Eine oder mehrere getrennte JSONL-Ergebnisdateien",
    )
    recommend.add_argument(
        "--re-evaluate",
        action="store_true",
        help="Gespeicherte Outputs und Agent-Traces mit der aktuellen Bewertungslogik neu auswerten",
    )
    recommend.add_argument(
        "--profile",
        "--objective-profile",
        dest="objective_profile",
        default="quality-first",
        help="Zielprofil aus profiles/ (z.B. quality-first, interactive, resource-constrained)",
    )
    recommend.add_argument(
        "--profiles-dir",
        type=Path,
        default=ROOT,
        help="Verzeichnis mit profiles/ und Zielprofilen",
    )
    recommend.set_defaults(func=cmd_recommend)

    sanitize = subparsers.add_parser(
        "sanitize",
        help="JSONL-Ergebnisse für eine öffentliche Weitergabe anonymisieren",
    )
    sanitize.add_argument("--input", type=Path, required=True, help="Private JSONL-Quelldatei")
    sanitize.add_argument("--output", type=Path, required=True, help="Öffentliche JSONL-Zieldatei")
    sanitize.add_argument(
        "--host-id",
        required=True,
        help="Neutrale öffentliche Host-ID, z.B. cpu-reference",
    )
    sanitize.add_argument(
        "--keep-timestamps",
        action="store_true",
        help="Zeitstempel bewusst in der öffentlichen Datei behalten",
    )
    sanitize.add_argument(
        "--keep-kernel",
        action="store_true",
        help="Kernelversion bewusst in der öffentlichen Datei behalten",
    )
    sanitize.add_argument(
        "--force",
        action="store_true",
        help="Vorhandene Zieldatei bewusst überschreiben",
    )
    sanitize.set_defaults(func=cmd_sanitize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
