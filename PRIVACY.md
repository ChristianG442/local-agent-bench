# Privacy and safe result publication

The benchmark is designed for local execution, but raw JSONL files contain
host metadata. Review every result before publishing it.

## Never publish

- API keys, tokens, passwords, cookies or connection strings
- private DNS names, IP addresses or runtime endpoints
- employee, customer or device-owner names
- serial numbers, asset tags or internal repository paths
- prompts or model outputs containing private production data
- unreviewed logs, attachments or complete workspace exports

The benchmark fixtures shipped in this repository are synthetic. Do not replace
them with real customer, employee or operational data in a public run.

## Safe export command

Create a separate publication copy:

```bash
lab sanitize \
  --input results/private-run.jsonl \
  --output results/public-run.jsonl \
  --host-id cpu-reference
```

By default this command:

- replaces the original host ID,
- replaces all run IDs,
- removes the Ollama runtime endpoint,
- removes the kernel version,
- removes exact timestamps and
- adds a `publication.sanitized=true` marker.

CPU/GPU model, thread count, RAM/VRAM, Ollama version, model digest,
quantization, benchmark configuration, measurements and evaluations remain
available because they are necessary to reproduce and interpret results.

Use `--keep-kernel` or `--keep-timestamps` only after an explicit publication
review. Existing output files are not overwritten unless `--force` is supplied.

## Publication checklist

1. Run `lab sanitize`; never edit the private source JSONL in place.
2. Search the public copy for internal hostnames, domains and user names.
3. Confirm that prompts and outputs contain synthetic benchmark data only.
4. Verify that the declared hardware matches the machine that produced the run.
5. Keep private and public files in separate directories.
6. Commit only an explicit whitelist of reviewed files.

An anonymized host is still a real host measurement. Do not relabel results
from another machine or generate synthetic performance values.