# Contributing

Contributions that improve reproducibility, platform coverage or agent
evaluation are welcome.

## Principles

- Keep all built-in tool fixtures deterministic and free of real side effects.
- Do not add private prompts, logs, endpoints or host identifiers.
- Distinguish model capability from agent-scaffold and host-performance effects.
- Prefer bounded, machine-checkable evaluators over unrestricted
  LLM-as-a-judge scoring for hard gates.
- Add regression tests for every scoring or contract change.
- Preserve raw JSONL records; new evaluators should use re-evaluation instead
  of rewriting source evidence.
- Change targets or hard gates only when repeated real measurements support it.

## Development

```bash
cd local-agent-bench
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Before submitting a change:

1. run all tests,
2. run `git diff --check`,
3. update `TESTS.md` when cases or scoring change,
4. update the benchmark and Hermes contract versions when semantics change and
5. ensure no JSONL, secrets, private hostnames or workspace files are included.

## Adding a benchmark case

Document:

- the capability being measured,
- why it matters for agents,
- expected tool sequence and state transition,
- deterministic pass/fail criteria,
- safety impact and
- known limitations.

Security-critical behavior must fail closed. A prompt-injection result may only
be scored as assessed when the untrusted fixture actually reached the model.