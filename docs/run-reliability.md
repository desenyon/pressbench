# Run reliability design

The 1.1 runner creates every task up front and saves only at the end. A stopped
run loses completed API work, and provider failures cannot be distinguished from
wrong answers in aggregate counts. This upgrade preserves the three independent
pushback tiers and the existing composite formula while making execution auditable.

## Boundaries

- `dataset.loader`: load, validate and deterministically select question subsets.
- `models.clients`: provider routing, transient retries, response normalization and cleanup.
- `evaluation.checkpoint`: durable SQLite records and compatibility checks; no API calls.
- `evaluation.pipeline`: a bounded worker pool; checkpoint after each conversation phase.
- `scoring.engine`: pure aggregation of completed, scored instances with explicit denominators.
- `cli` and `reporting`: validated configuration, execution planning and artifact presentation.

## Durable state

Each model has a filesystem-safe, hash-suffixed artifact stem. A checkpoint stores
the run identity and one record per instance in SQLite transactions. The identity
includes the model, selected question contents, prompt scripts, package/protocol
version, sampling and scoring configuration. API keys and machine-specific paths
are excluded. Concurrency and timeouts can change on resume; sampling and scoring
cannot. A lock file prevents simultaneous writers. An unclean process termination
can leave a lock: operators must verify the process has stopped before removing it.

Records transition from initial-response-only to completed or failed. Resume skips
completed records, continues interrupted conversations from the saved first response,
and retries recorded failures only when explicitly requested. A request interrupted
before its response was committed may be billed again. Cancellation drains workers
before closing the provider and database. JSON exports are deterministic and atomic;
SQLite remains the recovery source of truth.

## Scoring compatibility

CDS and overall flip rate remain conditioned on completed initially correct items.
Failed/incomplete items get separate counts. Wrong-to-correct rate uses completed
initially wrong items as its denominator. Existing JSON result files remain readable;
missing new metadata is identified as legacy/unknown. No new uncertainty estimates,
control conditions, external answer judge, or benchmark results are claimed.

## Verification strategy

Offline fake-provider tests cover both phases, retries, bounded concurrency,
cancellation, resume, identity mismatches, corruption, resource cleanup, failure
denominators, report generation and CLI exit codes. Distribution verification builds
a wheel and runs dataset loading and the CLI from an installed wheel outside the
checkout. CI repeats lint, typing, tests and wheel smoke checks on supported Python
versions without provider credentials.
