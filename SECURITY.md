# Security policy

This project executes LLM requests against a configured Ollama endpoint.
Built-in Hermes tools are mocks and must never access the real filesystem,
terminal, browser, network or persistent memory.

## Reporting a vulnerability

Do not include secrets, private host details, exploit payloads against real
systems or sensitive logs in a public issue. Use GitHub's private vulnerability
reporting or Security Advisory mechanism for the repository.

Include:

- affected version,
- minimal synthetic reproduction,
- expected and observed behavior and
- potential impact.

## Scope

Security issues include:

- a mock tool causing a real side effect,
- command, path or network execution reachable from benchmark fixtures,
- accidental secret or endpoint disclosure,
- unsafe JSONL publication behavior and
- critical safety gates being bypassed by malformed records.

Model failures observed in a benchmark are evaluation results, not by
themselves vulnerabilities in this project.