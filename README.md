# ModelRouter and RSL

ModelRouter is an OpenAI-compatible gateway for routing chat requests across
multiple model providers. It combines provider and key management, access
control, rate limiting, health-aware selection, keyword and semantic routing,
cascade and ensemble execution, usage accounting, and an administrator API.

RSL (Recursive Self-Learning) is the gateway's experimental, operator-controlled
adaptive learning subsystem. It records privacy-reduced routing telemetry,
accepts authenticated feedback, trains candidate routing policies offline, and
supports staged rollout with explicit rollback. RSL does not train foundation
model weights and does not autonomously publish or activate policies.


## Contents

- [What it provides](#what-it-provides)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Gateway operation](#gateway-operation)
- [API reference](#api-reference)
- [RSL adaptive learning system](#rsl-adaptive-learning-system)
- [Operations and safety](#operations-and-safety)
- [Development](#development)
- [Repository layout](#repository-layout)
- [License](#license)

## What it provides

### ModelRouter gateway

- **OpenAI-compatible serving:** expose `/v1/chat/completions` and
  `/v1/models` to clients while routing to configured provider adapters.
- **Provider abstraction:** configure OpenAI-compatible, Anthropic, and
  DeepSeek-compatible providers with separate base URLs, keys, retries, and
  rate limits. The provider registry is extensible.
- **Policy-aware selection:** route by explicit routing key or canonical model
  name, with provider preference and cost, latency, and quality biases.
- **Request controls:** enforce API-key access, model allow-lists, RPM limits,
  routing profiles, and internal request context headers.
- **Adaptive strategies:** add keyword rules, optional semantic intent routing,
  cascade escalation, and ensemble judging without changing the client API.
- **Operations:** inspect routes, synchronize remote models, check provider
  health, manage keys and models, and review usage through the Admin API.

### RSL adaptive learning

- **Reduced-content telemetry:** record task and request-shape features rather
  than prompt or completion bodies in RSL decision records.
- **Feedback-driven rewards:** combine available quality feedback, latency,
  estimated cost, and provider success into configurable rewards.
- **Conservative policy learning:** aggregate historical outcomes into a
  candidate policy and evaluate it on a chronological holdout when propensity,
  support, and sample-size requirements are met.
- **Guarded execution:** re-check the host routing policy, model availability,
  and estimated-cost limits before a learned action can be selected.
- **Explicit rollout:** move policies through `disabled`, `shadow`, `canary`,
  and `active` stages using administrator-authenticated requests.

## Architecture

The gateway loads provider and routing configuration, persists operational data
in SQLite, and sends each eligible chat request through the routing engine. The
RSL engine observes the baseline decision, filters safe candidate actions, and
only replaces the baseline when the current rollout stage permits it.

The [interactive RSL runtime architecture](./.archify/rsl-runtime.html) shows
the online decision path, persistence boundary, offline training loop, and
operator rollout controls. A static preview is available here:

![RSL runtime architecture](./.archify/rsl-dark.svg)

The main components are:

| Component | Responsibility |
| --- | --- |
| FastAPI application | Serves health, chat, model, feedback, and admin endpoints. |
| Router engine | Resolves model targets and applies provider, profile, and access rules. |
| Provider adapters | Translate gateway requests, rotate keys, retry, and normalize responses. |
| SQLite database | Stores providers, models, keys, routing rules, usage, and RSL state. |
| Cascade and ensemble engines | Escalate failed or low-quality responses, or judge multiple responses. |
| RSL engine | Makes bounded online decisions and persists decision metadata. |
| RSL trainer and evaluator | Builds candidate policies and estimates offline performance. |
| Rollout controller | Applies revision-checked shadow, canary, active, and rollback changes. |

## Quick start

### Prerequisites

- Python 3.10 or newer with a virtual-environment tool.
- Credentials and base URLs for at least one supported upstream provider.
- A writable working directory for `router.db` and optional embedding caches.

Runtime dependencies are listed in `requirements.txt`.

### Install

From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and replace every placeholder secret and URL.
Copy [`config/providers.example.yaml`](./config/providers.example.yaml) to
`config/providers.yaml`, then configure its provider, model, and pricing
entries. The local `config/providers.yaml` is ignored by Git because it may
contain credentials.

Initialize the database from YAML, then start the API:

```bash
python -m app.manage init
python -m app.main
```

The service listens on `http://localhost:8000` by default. Confirm startup
before sending traffic:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

The `init` command is idempotent for a non-empty database. Use a new
`ROUTER_DB` path or the interactive `reset` command when deliberately
re-seeding a development database.

## Configuration

### Environment variables

`.env.example` is the canonical starter file. The most important settings are:

| Variable | Purpose | Default or example |
| --- | --- | --- |
| `ROUTER_RUNTIME_MODE` | Select standalone or internal deployment behavior. | `standalone` |
| `ROUTER_DB` | SQLite database path. | `router.db` |
| `ROUTER_API_KEY` | Standalone client key used by the configured auth flow. | Set a secret |
| `ADMIN_API_KEY` | Bearer token for all admin endpoints. | Set a separate secret |
| `ROUTER_ENABLE_ADMIN_API` | Enable the `/admin` API. | `true` |
| `ROUTER_ENABLE_USAGE_API` | Enable usage endpoints and accounting. | `true` |
| `ROUTER_ENABLE_STANDALONE_KEYS` | Enable standalone API-key resolution. | `true` |
| `GATEWAY_BASE_URL` | Fallback base URL for provider families. | Provider URL |
| `*_BASE_URL` | Per-family provider base URL override. | Optional |
| `*_KEY_1` | Provider credential; additional numbered keys are supported. | Secret |
| `AUTO_SYNC_PROVIDER_MODELS` | Refresh provider model lists at startup when enabled. | `true` |
| `ROUTER_RSL_RECORDING_ENABLED` | Record eligible RSL decisions and outcomes. | `false` |
| `ROUTER_RSL_EPSILON` | Exploration probability for eligible policy actions. | `0.0` |
| `ROUTER_RSL_MAX_COST_RATIO` | Candidate estimated-cost ratio versus baseline. | `1.5` |
| `ROUTER_RSL_MAX_ESTIMATED_COST` | Optional absolute estimated cost ceiling per request in USD. | Unset |
| `ROUTER_RSL_ALLOWLIST` / `ROUTER_RSL_DENYLIST` | Optional comma-separated routing-key candidate constraints. | Empty |
| `ROUTER_RSL_STATE_TTL_SECONDS` | Rollout-state cache lifetime in a worker. | `5.0` |
| `ROUTER_RSL_CANARY_SALT` | Salt for stable canary subject bucketing. | `rsl-canary` |
| `ROUTER_RSL_DEFAULT_COMPLETION_TOKENS` | Completion-token estimate when the request omits a limit. | `512` |

Never commit `.env`, provider credentials, database files, or generated
embedding caches. Keep `ADMIN_API_KEY` separate from client and upstream keys.

### Provider YAML

Each provider has a `base_url`, optional health URL and timeout, a retry and
key-selection strategy, one or more keys, and one or more model definitions.
A model's `routing_keys` are the stable names clients use in the `model` field.
Pricing is an estimate used for routing and RSL guardrails; it is not a
provider billing guarantee.

Environment substitutions use `${NAME}` syntax. The loader applies a small set
of compatibility aliases, including `GATEWAY_BASE_URL`, `OPENAI_BASE_URL`,
`ANTHROPIC_BASE_URL`, `DEEPSEEK_BASE_URL`, and `GLM_BASE_URL`.

Model synchronization removes configured routes only when their upstream model
is no longer advertised. It preserves existing routing-key aliases, configured
prices, and enabled/disabled state. Newly discovered models use their upstream
name as the routing key when available; if another provider already owns that
key, the route is namespaced with the provider name. Local price-table data is
applied only to newly discovered routes, so syncing does not overwrite
operator-configured prices.

### Optional routing features

- **Keyword routing** matches literal text or regular expressions and targets a
  configured routing key. Configure rules through the admin endpoints.
- **Semantic routing** classifies request text with the configured embedding
  model and category definitions. Install the optional
  `sentence-transformers` dependency before enabling it.
- **Routing profiles** provide per-user or per-key defaults, allowed models,
  provider preferences, and selection biases.
- **Cascade rules** define an ordered chain and escalation or validation
  behavior. **Ensemble rules** can run participants and select a judge result.

Preview a route without sending it upstream:

```bash
python -m app.manage preview-route openai/gpt-4o-mini "Summarize this text"
```
Route previews resolve `auto` and profile defaults before applying
recommendations, matching the model-selection order used for chat requests.

## Gateway operation

### Request flow

1. Authenticate the caller and resolve request context, profile, and allowed
   models.
2. Resolve the requested model or default route from the database-backed model
   map.
3. Apply keyword, semantic, cascade, or ensemble behavior when enabled.
4. Select a provider key, enforce rate limits, and call the upstream adapter.
5. Normalize the response, record usage and outcome, and return an
   OpenAI-compatible response or stream.

OpenAI-compatible streams request usage-bearing terminal chunks when the
upstream supports them. The usage chunk is forwarded to the client and counted
against the selected provider key's TPM limit.

The gateway preserves the baseline route for supported fallback paths. Streaming
cascade and streaming ensemble execution are not integrated with RSL; ordinary
streaming can fall back only before its first content chunk.

Cascade judge models are subject to the same effective model permissions as
the cascade chain. A disallowed judge is rejected before any upstream request.
If every cascade provider attempt fails, non-streaming requests return HTTP 502
and record a failed cascade; streaming requests report an in-band SSE error
because the HTTP response has already started. Upstream SSE error events and
incomplete streams are treated as failures, not successful completions.

### Authentication and request context

Standalone clients normally send an API key in a Bearer authorization header.
Admin requests must send `Authorization: Bearer <ADMIN_API_KEY>`. Internal
integrations may supply trusted context headers such as `X-User-Id`,
`X-Subject-Id`, `X-Routing-Profile-Id`, `X-Allowed-Models`, and `X-Request-Id`.
Only use those internal headers behind a trusted boundary.

Example chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### CLI administration

The management module provides the following commands:

| Command | Use |
| --- | --- |
| `init` | Migrate and seed the database from `config/providers.yaml`. |
| `reset` | Delete and re-seed the database after confirmation. |
| `sync-models` | Discover provider models; optionally pass a provider name. |
| `check-health` | Check all providers or one named provider. |
| `preview-route` | Explain route selection for a model and prompt. |
| `gen-key` | Create a standalone user API key interactively. |
| `list-keys` | List key suffixes and status without printing secrets. |
| `list-models` | List configured routes and provider mappings. |
| `list-rules` | List keyword rules. |
| `list-intents` | List semantic categories from provider YAML. |
| `stats` | Print aggregate usage statistics. |
| `rsl-train` | Build and evaluate an RSL candidate policy. |

## API reference

All paths below are relative to the service origin. Unless noted otherwise,
admin routes require `Authorization: Bearer <ADMIN_API_KEY>` and return JSON.

### Public and client endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Report database, provider, model, and feature health. |
| `GET` | `/v1/models` | List available client-visible model routes. |
| `POST` | `/v1/chat/completions` | Create a routed chat completion or stream. |
| `POST` | `/v1/feedback` | Submit authenticated quality feedback to RSL. |

### Administrative gateway endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/admin/status` | Show provider and gateway status. |
| `GET` | `/admin/providers` | List configured providers. |
| `GET` | `/admin/providers/{name}` | Inspect one provider. |
| `POST` | `/admin/providers` | Create a provider. |
| `PUT` | `/admin/providers/{name}` | Update a provider. |
| `DELETE` | `/admin/providers/{name}` | Remove a provider. |
| `POST` | `/admin/providers/{name}/keys` | Add a provider key. |
| `POST` | `/admin/providers/sync-models` | Synchronize all provider models. |
| `POST` | `/admin/providers/check-health` | Check all provider health URLs. |
| `GET` | `/admin/models` | List database model routes. |
| `POST` | `/admin/models` | Create a model route. |
| `PUT` | `/admin/models/{model_id}` | Update a model route. |
| `DELETE` | `/admin/models/{model_id}` | Disable or remove a model route. |
| `GET` | `/admin/api-keys` | List client key metadata. |
| `POST` | `/admin/api-keys` | Create a client key. |
| `DELETE` | `/admin/api-keys/{key_id}` | Revoke a client key. |
| `GET` | `/admin/usage` | Query usage records. |
| `GET` | `/admin/usage/summary` | Return aggregate usage. |
| `GET` | `/admin/usage/daily` | Return daily usage aggregates. |
| `POST` | `/admin/route/preview` | Preview a route with request context. |
| `GET` | `/admin/keyword-rules` | List keyword routing rules. |
| `POST` | `/admin/keyword-rules` | Create a keyword routing rule. |
| `PUT` | `/admin/keyword-rules/{rule_id}` | Update a keyword routing rule. |
| `DELETE` | `/admin/keyword-rules/{rule_id}` | Delete a keyword routing rule. |
| `GET` | `/admin/routing-profiles` | List routing profiles. |
| `POST` | `/admin/routing-profiles` | Create a routing profile. |
| `PUT` | `/admin/routing-profiles/{profile_id}` | Update a routing profile. |
| `DELETE` | `/admin/routing-profiles/{profile_id}` | Delete a routing profile. |
| `GET` | `/admin/cascade/rules` | List cascade rules. |
| `POST` | `/admin/cascade/rules` | Create a cascade rule. |
| `GET` | `/admin/cascade/stats` | Inspect cascade outcomes. |

Use the FastAPI OpenAPI document served by the running application for exact
request and response schemas. The endpoint list above intentionally omits
less common update and detail variants that are discoverable there.

## RSL adaptive learning system

RSL is an auditable policy experiment loop layered on top of the gateway. It
learns routing preferences from observed outcomes, not model weights. Training
and rollout are separate operator actions.

### Lifecycle

1. **Record:** when enabled, eligible requests create a reduced-content decision
   record with task, shape, baseline, candidate actions, and propensities.
2. **Execute:** the rollout stage determines whether the baseline or candidate
   action serves the request. Existing access checks remain authoritative.
3. **Observe:** usage, latency, success or failure, estimated cost, and available
   feedback are joined to outcomes.
4. **Train:** `rsl-train` aggregates rewards into a candidate policy and writes
   evaluation metadata. It does not activate the policy.
5. **Review:** an operator inspects status, policy details, preview output, and
   evaluation limitations.
6. **Roll out:** an operator chooses shadow, canary, or active and can roll back
   or disable the policy at any time.

### Decision and safety behavior

RSL candidates are filtered by the host's access permissions, model
availability, and estimated-cost guardrails. Candidate estimates are not
provider-billed cost caps. Stable canary bucketing uses the configured subject
identifier and salt. Epsilon-greedy exploration logs the exact action
probability distribution used for a decision.

RSL integration is bounded by request mode:

- Non-streaming ordinary requests can use an RSL replacement and supported
  failure paths can try the original baseline.
- Non-streaming cascade requests can select within the resolved chain.
- Non-streaming ensemble requests can select the judge.
- Streaming cascade and streaming ensemble are not integrated with RSL.
- Streaming fallback can switch only before the first content chunk.

RSL decision features intentionally omit prompt and completion bodies, but the
host application and upstream providers may have separate logs. Review their
retention and access controls independently.

### Enable recording

Start with recording enabled and rollout disabled. Add these settings to
`.env`, then restart the service. This collects evidence without changing the
model selected for traffic:

```dotenv
ROUTER_RSL_RECORDING_ENABLED=true
ROUTER_RSL_EPSILON=0.0
ROUTER_RSL_MAX_COST_RATIO=1.5
ROUTER_RSL_STATE_TTL_SECONDS=5.0
ROUTER_RSL_CANARY_SALT=replace-with-a-deployment-specific-salt
```

Restart workers after changing environment variables. RSL rollout state is
cached, so a deployed state change can take up to the configured TTL to reach a
worker.

### Submit feedback

Feedback is tied to the authenticated subject on the original decision. A
caller cannot submit feedback for a different subject. Ratings are normalized
quality signals; categorical labels can also be supplied.
When RSL recording is enabled, successful chat-completion responses include
the persisted decision identifier at `rsl.request_id`, even while rollout is
disabled. Submit that identifier as `request_id` below; identifiers for
decisions that could not be persisted are not returned.

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "rating": 1.0,
    "label": "good"
  }'
```

### Train and evaluate a candidate

Training uses a maturation window so late feedback can arrive. It is
baseline-only in the current implementation and produces a candidate for
review rather than a live policy:

```bash
python -m app.manage rsl-train \
  --window-days 7 \
  --feedback-wait-hours 24 \
  --min-samples-per-cell 30 \
  --evaluation-fraction 0.25 \
  --min-effective-samples 30 \
  --min-independent-clusters 30
```

The result includes a policy version, dataset identifier, training and
holdout counts, and evaluation status. An evaluation may be
`not_evaluable` when action support, independent clusters, effective sample
size, or other causal requirements are not met.

### Roll out and roll back

RSL admin endpoints are mounted beneath `/admin/rsl` and use the same admin
Bearer token as the gateway admin API.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/admin/rsl/status` | Show rollout, settings, and decision summary. |
| `GET` | `/admin/rsl/policies` | List candidate and historical policies. |
| `GET` | `/admin/rsl/policies/{version}` | Inspect one policy and its metrics. |
| `POST` | `/admin/rsl/preview` | Preview an RSL decision without upstream execution. |
| `POST` | `/admin/rsl/rollout/shadow` | Record a proposal while executing baseline. |
| `POST` | `/admin/rsl/rollout/canary` | Serve a policy to a percentage of stable subjects. |
| `POST` | `/admin/rsl/rollout/activate` | Serve the policy to eligible managed traffic. |
| `POST` | `/admin/rsl/rollout/rollback` | Return to the previous policy or disable. |
| `POST` | `/admin/rsl/rollout/disable` | Disable learned routing. |
| `GET` | `/admin/rsl/audit` | Review rollout transitions and operators. |

Example staged rollout:

```bash
curl -X POST http://localhost:8000/admin/rsl/rollout/shadow \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"policy_version": "rsl-candidate-a1b2c3d4e5f6"}'

curl -X POST http://localhost:8000/admin/rsl/rollout/canary \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"policy_version": "rsl-candidate-a1b2c3d4e5f6", "percent": 10.0}'

curl -X POST http://localhost:8000/admin/rsl/rollout/activate \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"policy_version": "rsl-candidate-a1b2c3d4e5f6"}'
```

Rollback is explicit and revision checked:

```bash
curl -X POST http://localhost:8000/admin/rsl/rollout/rollback \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"note": "Provider latency regression detected"}'
```

Policies without supported causal evaluation are blocked by default for
canary and active rollout. An authorized operator can explicitly acknowledge
an unproven evaluation only when the operational risk is understood.

### RSL storage and migrations

RSL uses `rsl_*` tables in the configured SQLite database. Schema migrations
are incremental and idempotent, and preserve existing records while adding
required columns. Back up the database before upgrading a deployment. Rollout
transitions use revision checks and SQLite immediate transactions so stale or
concurrent transitions are rejected instead of silently overwriting state.

RSL stores subject identifiers to enforce feedback ownership and stable
canary bucketing. It currently has no automated data-retention cleanup or
subject-identifier pseudonymization. Treat the database and its backups as
sensitive, and define access, retention, and deletion procedures before
collecting production telemetry.

## Operations and safety

- Use separate secrets for gateway clients, provider keys, and administrators.
- Put the admin API behind a private network, access proxy, or equivalent
  boundary; do not expose it as a public client endpoint.
- Treat pricing and latency values as routing estimates and monitor provider
  invoices and service-level metrics separately.
- Start RSL in recording mode, validate data quality, and use shadow before
  canary. Define rollback ownership and an incident procedure first.
- Back up `router.db` before schema migrations, reset operations, or deployment
  upgrades. Never use `reset` on a production database.
- Inspect provider and application logs for prompt and completion retention,
  because RSL's reduced-content feature records do not cover every log path.
- Restrict internal context headers to trusted infrastructure; clients must not
  be allowed to self-assert model permissions or subject identity.

## Development

Useful checks while developing include:

```bash
python -m app.manage preview-route openai/gpt-4o-mini "test prompt"
python -m app.manage list-models
python -m app.manage stats
```

The repository does not currently provide a production deployment manifest.
Supply process supervision, TLS termination, secret management, backups, and
observability appropriate to the deployment environment.

## Repository layout

```text
.
├── app/
│   ├── api/                 # Chat, model, and administrator HTTP endpoints
│   ├── cascade/             # Response validation and webhook or local judges
│   ├── core/                # Database, router, policies, health, and limits
│   ├── providers/           # Upstream provider adapters and key pools
│   └── schemas/             # Pydantic request and response models
├── rsl/
│   ├── api/                 # Feedback and administrator RSL endpoints
│   ├── collector.py         # Reduced-content feature and outcome collection
│   ├── engine.py            # Online decisions and policy cache
│   ├── evaluator.py         # Chronological holdout evaluation
│   ├── guardrails.py        # Candidate access and cost filtering
│   ├── rollout.py           # Staged rollout and stable canary bucketing
│   ├── scoring.py           # Multi-objective reward calculation
│   ├── store.py             # RSL persistence and migrations
│   ├── trainer.py           # Offline candidate policy training
├── .archify/                # Generated RSL architecture assets
├── .env.example             # Environment configuration template
├── config/
│   └── providers.example.yaml  # Safe provider configuration template
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT license
```

## License

This project is available under the [MIT License](./LICENSE).

## Acknowledgements

The architecture diagrams were created with the Archify skill using
[Archify](https://github.com/tt-a1i/archify).
