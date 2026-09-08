# Warum ein gutes Sprachmodell noch kein guter Agent ist

## Ein lokaler Benchmark zwischen Qualität, Tool-Nutzung, Sicherheit und Hardware

Wenn über lokale Sprachmodelle gesprochen wird, dominieren häufig drei Zahlen:
Parameter, Quantisierung und Tokens pro Sekunde. Für einen produktiven
KI-Agenten reichen diese Angaben nicht.

Ein Agent muss nicht nur eine plausible Antwort formulieren. Er muss erkennen,
ob überhaupt ein Tool nötig ist, fehlende Angaben erfragen, strukturierte
Argumente erzeugen, Tool-Ergebnisse weiterverarbeiten und über mehrere Schritte
einen konsistenten Zustand halten. Gleichzeitig darf er unsichere Anweisungen
aus Tool-Ausgaben nicht übernehmen. Und all das muss auf der vorhandenen
Hardware innerhalb eines brauchbaren Zeit- und Speicherbudgets funktionieren.

Aus dieser Fragestellung ist **Local Agent Bench** entstanden: ein portabler,
deterministischer Benchmark für lokale Ollama-Modelle.

## Was der Benchmark anders macht

Der Test trennt fünf Ebenen:

1. **Deutsches Sprach- und Aufgabenverständnis**  
   Rechnen, Formatdisziplin, strukturierte Rechnungsextraktion, Negation und
   Faktentreue bei Zusammenfassungen.

2. **Native Tool Calls**  
   Auswahl des richtigen Tools, korrekte Argumente und vor allem bewusster
   Verzicht auf einen Tool Call, wenn Informationen fehlen oder bereits
   vorliegen.

3. **Mehrstufige Agentenschleifen**  
   Tool-Ergebnis verarbeiten, den nächsten Tool Call tatsächlich ausführen,
   Fehler korrigieren, Zustand speichern und wieder auslesen.

4. **Sicherheit**  
   Destruktive Aufforderungen ablehnen und eingebettete Prompt-Injections in
   nicht vertrauenswürdigen Tool-Ausgaben ignorieren.

5. **Reale Hosteigenschaften**  
   Cold Load, Time to First Token, Wall Time, Tokens pro Sekunde, RAM und – auf
   NVIDIA-Systemen – VRAM.

Die Tools sind sichere In-Memory-Simulationen. Es werden keine echten Dateien
verändert, keine Terminalbefehle ausgeführt und keine Webaktionen ausgelöst.
Damit lässt sich das Verhalten reproduzierbar testen, ohne dem Modell reale
Side Effects zu erlauben.

## Warum ein einzelner Tool-Test nicht genügt

Eine der interessantesten Beobachtungen entstand bei der Analyse eines
mittelgroßen lokalen Modells. Einfache Tool Calls funktionierten zuverlässig.
Auch ein kontrollierter Retry nach einem temporären Fehler war möglich.

In mehrstufigen Memory-Fällen zeigte sich jedoch ein anderes Muster:

> Das Modell führte den ersten Tool Call aus, beschrieb den nächsten Schritt
> anschließend sprachlich – rief das dafür notwendige Tool aber nicht auf.

Die Antwort klang überzeugend: Ein Wert sei gespeichert oder wieder ausgelesen
worden. Im Agententrace fehlte die Aktion jedoch.

Für einen Chatbot ist das möglicherweise nur eine ungenaue Formulierung. Für
einen Agenten ist es ein fundamentaler Unterschied. Eine behauptete Aktion ist
kein ausgeführter Systemzustand.

Deshalb bewertet der Benchmark Toolfolge, Argumente, Schleifenabschluss,
Zustand und finale Aussage getrennt.

## Auch der Benchmark selbst muss getestet werden

Die vollständigen Traces zeigten nicht nur Modellfehler, sondern auch
Schwächen in der ersten Benchmarkversion:

- Eine Suche nach „Notiz Projekt Nordstern“ wurde abgelehnt, weil exakt
  „Projekt Nordstern“ erwartet wurde.
- Ein Modell korrigierte eine Web-Suchquery korrekt, der zweite Call wurde aber
  fälschlich als unerwartetes Tool behandelt.
- Eine ausdrückliche Frage nach „Uhrzeit oder Zeitfenster“ scheiterte, weil der
  Evaluator wörtlich das Wort „wann“ erwartete.

Diese Fälle wurden nicht genutzt, um ein Modell nachträglich schönzurechnen.
Stattdessen wurde der Testvertrag korrigiert:

- informationsgleiche Sucherweiterungen werden akzeptiert,
- Argumentkorrekturen funktionieren in allen Fällen konsistent und
- notwendige Rückfragen werden semantisch bewertet.

Echte Fehler bleiben echte Fehler. Wenn `memory_set` oder `memory_get` nicht
aufgerufen wurde, kann eine sprachliche Behauptung den fehlenden Tool Call
nicht ersetzen.

## Erste reale CPU-Ergebnisse

Eine vollständige Baseline wurde auf einem anonymisierten CPU-Referenzhost
ausgeführt:

- HP ProDesk 600 G3 DM
- Intel Core i5-7500T
- vier CPU-Threads
- 32-GB-RAM-Klasse
- CPU-only
- Ollama 0.33.2
- Qwen3.5 4B, Q4_K_M
- 8.192 Kontexttokens
- KV-Cache `f16`
- Thinking deaktiviert
- 101 reale JSONL-Records

Die dokumentierte Neuauswertung der unveränderten Rohdaten ergab:

| Zielprofil | Zielscore | Qualität | Hard Gates |
|---|---:|---:|---|
| Quality First | 76 % | 80 % | bestanden |
| Interactive | 55 % | 80 % | TTFT-Gate nicht bestanden |
| Resource Constrained | 85 % | 80 % | bestanden |

Das ist kein Widerspruch. Dasselbe Modell kann für einen ressourcenschonenden
Always-on-Betrieb geeignet und für eine interaktive Nutzung zu langsam sein.
Genau deshalb werden Qualitäts-, Latenz- und Ressourcenziele nicht zu einer
einzigen universellen Rangliste vermischt.

Die Grenzwerte wurden nach diesem Lauf nicht gelockert. Das Interactive-Gate
trennt den langsamen CPU-Track wie vorgesehen. Eine vollständige reale
12-GB-GPU-Baseline steht noch aus; bis dahin wäre eine plattformübergreifende
Kalibrierung verfrüht.

## Was ich aus dem Experiment mitnehme

### 1. Agentenqualität ist Zustandsqualität

Entscheidend ist nicht nur, ob das Modell die richtige Aktion beschreiben kann.
Es muss den richtigen Zustandsübergang tatsächlich über das Tool-Protokoll
ausführen.

### 2. Restraint ist eine positive Fähigkeit

Ein Modell, das bei unvollständigen Angaben kein Tool aufruft, ist nicht
„weniger agentisch“. Es handelt kontrollierter.

### 3. Sicherheit muss beobachtbar sein

Wenn eine Prompt-Injection das Modell wegen eines früheren Toolfehlers nie
erreicht, ist der Test nicht bestanden und nicht fehlgeschlagen. Er ist
`not_assessed`. Alles andere würde fehlende Exposition mit erfolgreicher Abwehr
verwechseln.

### 4. Hardware gehört zum Agentenbenchmark

Eine Toolschleife multipliziert Inferenzlatenz. Ein Modell mit guter
Einzelantwort kann im mehrstufigen Ablauf unpraktisch werden. Cold Load, TTFT
und Speicherbedarf sind deshalb keine Nebensache.

### 5. Benchmarkfehler müssen versioniert korrigiert werden

Rohdaten bleiben unverändert. Neue Bewertungsregeln werden als neue
Benchmarkversion angewendet und die ursprüngliche Bewertung bleibt
nachvollziehbar.

## Öffentliche Weiterentwicklung

Der Benchmark wird als Open-Source-Werkzeug veröffentlicht. Eigene Hosts,
Modelle, Quantisierungen, KV-Cache-Typen und Zielprofile lassen sich über
TOML-Dateien konfigurieren.

Für veröffentlichte Ergebnisse gibt es einen eigenen Anonymisierungsschritt.
Interne Hostnamen, Endpoints, Kernelversionen, Run-IDs und exakte Zeitstempel
werden entfernt, während reproduktionsrelevante Hardware- und Messdaten
erhalten bleiben.

Methodisch orientiert sich das Projekt an Ideen aus BFCL, τ-bench, MINT und
AgentDojo, bleibt aber bewusst klein, lokal und deterministisch. Es ersetzt
keine vollständigen Browser-, Desktop- oder Software-Engineering-Benchmarks.

Mich interessiert besonders:

- Welche Agentenfähigkeit fehlt in diesem Katalog?
- Wie sollten Tool-Restraint und sichere Rückfragen gewichtet werden?
- Welche lokalen Modelle halten mehrstufige Tool-Loops zuverlässig durch?

Der nächste belastbare Meilenstein ist der Vergleich mit einer realen
12-GB-NVIDIA-Baseline.

---

**Hinweis für die Veröffentlichung:** Vor dem Posten Repository-URL,
Lizenzhinweis und gegebenenfalls einen Link zur Methodendokumentation ergänzen.