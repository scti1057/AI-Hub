# AI Hub

AI Hub is a private local multi-agent system that runs on a personal machine and is exposed in a controlled way through a FastAPI web app. The project is designed around a manager-driven workflow: the user talks to a manager, the manager plans and routes the request, and specialized workers handle bounded tasks inside explicit security limits.

This repository is both the application and the operating handbook for the current system state. It is meant to be the first document a new Codex chat, the project owner, or a technical reviewer should read.

## Project Overview

### What AI Hub is

AI Hub is a local/private orchestration platform for:

- a web-based chat interface
- persistent thread-based conversations
- a manager agent backed by Ollama
- bounded worker delegation
- approval-gated Python execution
- workspace-restricted coding tasks
- push notifications for approval events

The system is intentionally conservative. It is not trying to be a fully autonomous agent runtime. The current architecture prioritizes:

- traceability
- human approval before execution
- strict workspace boundaries
- incremental delegation instead of uncontrolled autonomy

### Vision

The long-term vision is a usable private multi-agent assistant where:

- the user primarily interacts with a manager agent
- the manager maintains context across threads
- coding and research workers operate in specialized roles
- risky actions require explicit human approval
- local models via Ollama provide natural language planning and responses
- the system remains safe enough to use from mobile devices

### Current Development Status

AI Hub is no longer a stub project. It is already a functioning local system with a real web app, persistent memory, mobile access, push notifications, approval gating, and a manager integrated with Ollama.

The system is in a strong “controlled alpha” phase:

- usable for real local workflows
- intentionally constrained
- good enough for iterative development and testing
- not yet a full autonomous agent platform

### What already works now

The following are implemented and testable today:

- private FastAPI web app
- session login
- persistent thread/chat model in SQLite
- manager workflow with Ollama plus safe fallback
- coding delegation with structured internal actions
- approval requests for Python execution
- execution after user approval
- workspace-restricted file operations
- sensitive path blocking
- push notifications
- logging/tracing across manager, execution, and API flows

## Core Capabilities

### User-facing capabilities

- Private web app with login
- Thread-based chat UX
- Mobile usage via iPhone home-screen web app
- Tailscale-based private remote access
- Push notifications when an approval is needed

### Manager capabilities

- Natural user interaction
- Thread-aware context handling
- Routing between direct answer, coding delegation, and research delegation
- Ollama-backed planning with safe fallback behavior
- Approval coordination
- Logging of routing, fallback, and delegation decisions

### Coding capabilities

- Per-thread coding workspace
- Structured internal coding actions:
  - `make_directory`
  - `create_file`
  - `read_file`
  - `delete_path`
  - `request_execution`
  - `list_files`
- Multi-step action batches
- Structured action results
- Approval request creation only after the required file exists

### Execution capabilities

- Python execution only after approval
- `bubblewrap` sandbox as preferred execution backend
- controlled fallback to subprocess mode when `bubblewrap` cannot be used
- workspace-only execution target validation

## Architecture

### High-level architecture

`User -> Web App/API -> Manager Workflow -> Routing/Planning -> Worker Delegation -> Tools -> Approval/Execution`

### Main components

#### User

The user interacts through the web app. The primary interface is a chat thread. The user can also approve or reject Python execution requests.

#### Web app / API

The web layer lives in [app.py](src/ai_hub/web/app.py) and serves:

- login/session handling
- static web UI
- thread APIs
- chat API
- approval APIs
- push subscription APIs

The frontend lives in [index.html](src/ai_hub/web/static/index.html) and the service worker in [sw.js](src/ai_hub/web/static/sw.js).

#### Manager

The manager runtime is implemented in [workflow.py](src/ai_hub/orchestration/workflow.py). It is the central orchestrator and is responsible for:

- storing user messages
- preparing manager planning input
- calling the Ollama-backed planner
- combining planner output with safe routing policy
- delegating to workers
- creating approval requests
- sending push notifications

The manager prompt is in [manager.txt](src/ai_hub/prompts/manager.txt).

#### Coding agent

The coding worker is implemented in [coding_agent.py](src/ai_hub/agents/coding_agent.py). It now runs on structured internal actions rather than directly on free-form tool parsing.

Its execution pipeline is:

1. receive a structured delegation plan if available
2. otherwise derive a structured action batch from text
3. execute actions in order
4. return structured execution results
5. optionally prepare an approval request

The coding prompt file exists in [coding_agent.txt](src/ai_hub/prompts/coding_agent.txt), but the current execution path is mostly server-side and structured rather than prompt-driven.

#### Research agent

The research worker is currently intentionally lightweight and read-oriented. It is implemented in [research_agent.py](src/ai_hub/agents/research_agent.py). At the moment it acts more as an analytical placeholder than a fully developed agent.

#### Structured schemas

Important internal schemas live in [src/ai_hub/schemas](src/ai_hub/schemas):

- [coding_actions.py](src/ai_hub/schemas/coding_actions.py)
- [coding_delegation.py](src/ai_hub/schemas/coding_delegation.py)
- [manager_plan.py](src/ai_hub/schemas/manager_plan.py)

These schemas are important because they define the structured contract between manager planning and coding execution.

#### Workspace tools

Low-level coding tools live in [file_tools.py](src/ai_hub/tools/file_tools.py). They enforce:

- per-thread workspace roots
- relative-path-only access
- path escape blocking
- sensitive path blocking

#### Approval and execution

Execution policy is implemented in [code_runner.py](src/ai_hub/tools/code_runner.py). This module validates Python execution requests, chooses the sandbox backend, and performs the actual execution only after approval.

#### Push notifications

Push behavior is implemented in [push_notify.py](src/ai_hub/tools/push_notify.py) and web push subscriptions are stored in SQLite.

#### Ollama integration

The manager’s Ollama integration lives in:

- [manager_planner.py](src/ai_hub/llm/manager_planner.py)
- [ollama_client.py](src/ai_hub/llm/ollama_client.py)
- [model_router.py](src/ai_hub/llm/model_router.py)

The manager uses Ollama for planning and drafting, but the backend still enforces safety decisions.

#### Logging

Central logging configuration is in [logging_config.py](src/ai_hub/logging_config.py). Logs are written to the `logs/` directory with rotating file handlers.

## Message / Request Flow

### Step-by-step flow

1. The user sends a chat message through the web app.
2. `POST /api/chat` in [app.py](src/ai_hub/web/app.py) forwards the request into the manager workflow.
3. The manager ensures a thread exists, stores the user message, and formats recent thread history.
4. The manager planner asks Ollama for a structured plan.
5. The workflow combines:
   - safe local routing policy
   - structured manager plan from Ollama
6. The result is one of:
   - direct manager reply
   - coding delegation
   - research delegation
7. If the route is `coding`, the manager tries to provide a structured `CodingDelegationPlan`.
8. The coding agent prefers that structured plan.
9. If no valid structured plan is available, the coding agent falls back to building a structured action batch from text.
10. The coding agent executes actions through the server-side tool layer.
11. If execution is requested, the coding agent prepares an approval request instead of running Python directly.
12. The manager stores the approval request and sends a push notification.
13. The user approves or rejects the execution in the web app.
14. If approved, the backend validates the command again and runs Python through `bubblewrap` when possible.
15. The result is stored in the thread and logged.

### Textual flow diagram

`User`
-> `FastAPI /api/chat`
-> `ManagerWorkflow`
-> `ManagerPlanner (Ollama + fallback)`
-> `Router + policy merge`
-> `DelegationService`
-> `CodingAgent / ResearchAgent`
-> `Workspace tools / approval system`
-> `Push + thread update`
-> `User`

### Approval flow

`Coding request`
-> `structured action batch`
-> `request_execution action`
-> `approval record created`
-> `push notification`
-> `user approves`
-> `execute_python_approval()`
-> `bubblewrap or subprocess fallback`
-> `result written back to thread`

## Security Model

Security is a first-class design goal in AI Hub.

### Workspace boundaries

Each thread gets its own coding workspace below:

`data/workspaces/coding_agent/<thread_id>/`

The coding tools only allow relative paths inside that workspace.

### Sensitive paths are blocked

The tool layer blocks requests that touch or try to traverse into sensitive locations such as:

- `.env`
- `.git`
- `.ssh`
- `secrets`
- `../`
- absolute paths like `/etc/passwd`
- `~` paths

The enforcement lives in [file_tools.py](src/ai_hub/tools/file_tools.py), not in the LLM.

### Approval before execution

The coding agent can prepare an execution request, but it cannot directly run Python without approval. The manager creates an approval record and the user must explicitly approve it.

### Execution sandbox

Preferred execution backend:

- `bubblewrap` (`bwrap`)

Fallback backend:

- subprocess fallback if `bubblewrap` is not available or cannot create the required namespace

This fallback is explicit and visible in the execution result and logs. It is not silent.

### What the LLM can do

The manager LLM can:

- understand requests
- summarize
- propose routing
- draft replies
- optionally produce a structured coding delegation plan

The LLM cannot:

- bypass workspace validation
- bypass approval checks
- force arbitrary file access
- directly execute Python outside backend policy

### Fallback behavior

If Ollama is unavailable, misconfigured, or returns an unusable plan:

- the manager falls back to deterministic backend logic
- routing still works
- safety rules remain unchanged

## Important Project Files and Directories

### Top level

- [README.md](README.md): project overview and onboarding
- [requirements.txt](requirements.txt): runtime Python dependencies
- [requirements-dev.txt](requirements-dev.txt): development dependencies
- [pyproject.toml](pyproject.toml): test path + formatting/lint settings

### `src/ai_hub/`

#### Core runtime

- [config.py](src/ai_hub/config.py): environment/config values
- [state.py](src/ai_hub/state.py): enums and shared state objects
- [logging_config.py](src/ai_hub/logging_config.py): centralized logging setup

#### Agents

- [agents/manager.py](src/ai_hub/agents/manager.py): simple manager entrypoint wrapper
- [agents/coding_agent.py](src/ai_hub/agents/coding_agent.py): structured coding action execution
- [agents/research_agent.py](src/ai_hub/agents/research_agent.py): current research placeholder

#### Orchestration

- [orchestration/workflow.py](src/ai_hub/orchestration/workflow.py): central manager workflow
- [orchestration/router.py](src/ai_hub/orchestration/router.py): conservative route heuristic
- [orchestration/delegation.py](src/ai_hub/orchestration/delegation.py): worker dispatch

#### Schemas

- [schemas/manager_plan.py](src/ai_hub/schemas/manager_plan.py): manager LLM plan model
- [schemas/coding_actions.py](src/ai_hub/schemas/coding_actions.py): internal coding actions
- [schemas/coding_delegation.py](src/ai_hub/schemas/coding_delegation.py): manager-to-coding structured handoff

#### Memory

- [memory/sqlite_db.py](src/ai_hub/memory/sqlite_db.py): SQLite persistence layer
- [memory/store.py](src/ai_hub/memory/store.py): higher-level store API
- [memory/history.py](src/ai_hub/memory/history.py): thread history formatting

#### LLM

- [llm/manager_planner.py](src/ai_hub/llm/manager_planner.py): Ollama-backed manager planner
- [llm/ollama_client.py](src/ai_hub/llm/ollama_client.py): low-level Ollama HTTP client
- [llm/model_router.py](src/ai_hub/llm/model_router.py): model mapping by role

#### Tools

- [tools/file_tools.py](src/ai_hub/tools/file_tools.py): workspace filesystem enforcement
- [tools/code_runner.py](src/ai_hub/tools/code_runner.py): execution validation and sandboxed execution
- [tools/push_notify.py](src/ai_hub/tools/push_notify.py): web push delivery

#### Web

- [web/app.py](src/ai_hub/web/app.py): FastAPI app
- [web/static/index.html](src/ai_hub/web/static/index.html): main frontend
- [web/static/login.html](src/ai_hub/web/static/login.html): login UI
- [web/static/sw.js](src/ai_hub/web/static/sw.js): service worker
- [web/static/manifest.webmanifest](src/ai_hub/web/static/manifest.webmanifest): installable app manifest

#### Prompts

- [prompts/manager.txt](src/ai_hub/prompts/manager.txt)
- [prompts/coding_agent.txt](src/ai_hub/prompts/coding_agent.txt)
- [prompts/research_agent.txt](src/ai_hub/prompts/research_agent.txt)

### `scripts/`

- [run_web.py](scripts/run_web.py): local web server entrypoint via Uvicorn
- [init_db.py](scripts/init_db.py): initialize SQLite DB
- [test_push.py](scripts/test_push.py): send a test web push notification
- [run_local.py](scripts/run_local.py): local manager test script
- [generate_vapid_keys.py](scripts/generate_vapid_keys.py): VAPID key generation

### `data/`

- `data/memory/ai_hub.db`: SQLite database
- `data/workspaces/coding_agent/`: per-thread workspaces

### `logs/`

- `logs/app.log`
- `logs/manager.log`
- `logs/execution.log`

## How to Start the Project

The commands below assume you are in the project root.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional dev dependencies:

```bash
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

Create a `.env` file with at least the relevant values for:

- `APP_USERNAME`
- `APP_PASSWORD`
- `SESSION_SECRET`
- `OLLAMA_BASE_URL`
- `MANAGER_MODEL`
- `VAPID_PUBLIC_KEY`
- `VAPID_PRIVATE_KEY_PATH`
- `VAPID_SUBJECT`
- `LOG_LEVEL`

The runtime config is defined in [config.py](src/ai_hub/config.py).

### 3. Initialize the database

```bash
./.venv/bin/python scripts/init_db.py
```

In practice the DB is also initialized automatically when the web app starts, but this command is useful for setup and debugging.

### 4. Start the web app

```bash
./.venv/bin/python scripts/run_web.py
```

This starts the app locally on:

`http://127.0.0.1:8000`

### 5. Run tests

```bash
./.venv/bin/python -m pytest -q
```

### 6. Compile-check the code

```bash
./.venv/bin/python -m compileall src tests
```

### 7. Test push notifications

```bash
./.venv/bin/python scripts/test_push.py
```

### 8. Check Ollama

Useful local checks:

```bash
curl http://localhost:11434/api/tags
ollama list
```

### 9. View logs

```bash
tail -n 80 logs/app.log
tail -n 80 logs/manager.log
tail -n 80 logs/execution.log
tail -f logs/app.log
```

### 10. Tailscale / private remote access

AI Hub is intended to be reachable privately over Tailscale. The exact Tailscale setup depends on the local machine, but a common pattern is:

```bash
tailscale status
tailscale serve status
```

If you expose the local FastAPI app with Tailscale Serve, point it at the local web port used by [run_web.py](scripts/run_web.py).

Because Tailscale configuration can differ per host, treat this README as describing the application side, not as the single source of truth for your machine-specific Tailscale setup.

## Logging / Debugging

### Log files

AI Hub writes rotating UTF-8 logs to:

- `logs/app.log`
- `logs/manager.log`
- `logs/execution.log`

### What each log contains

#### `logs/app.log`

Broad application events across:

- API requests
- manager workflow
- worker activity
- approvals
- execution
- push notifications

#### `logs/manager.log`

Manager-focused events:

- Ollama planner requests/responses
- manager routing decisions
- `source=ollama|fallback`
- `llm_decision`
- `final_decision`
- `fallback_reason`
- approval creation

#### `logs/execution.log`

Execution and tool details:

- coding action batches
- action execution order
- file operations
- blocked paths
- execution validation
- execution backend
- push delivery events

### Useful debug commands

```bash
tail -n 100 logs/app.log
tail -n 100 logs/manager.log
tail -n 100 logs/execution.log
grep "approval_" logs/app.log
grep "structured_plan_used" logs/manager.log
grep "sandbox_backend" logs/execution.log
```

### What to look for while debugging

If the manager behaves strangely:

- inspect `logs/manager.log`
- compare `llm_decision` vs `final_decision`
- check `source=ollama` vs `source=fallback`

If coding actions are wrong:

- inspect `logs/execution.log`
- look for `coding_action_batch`
- verify `structured_plan_used`, `fallback_to_heuristic`, `action_execution_order`

If approval/execution fails:

- inspect `approval_created`, `approval_approved`, `approval_execution_finished`
- check `sandbox_backend`
- inspect `returncode`, `stderr`, and blocked reasons

## Typical Workflows

### 1. Strategic question to the manager

Example:

`Wie würdest du ein kleines Python-Projekt strukturieren, das später über Tailscale und Web Push erreichbar ist?`

Expected behavior:

- manager handles this as a direct strategic request
- no coding workspace action is executed
- no approval is created

### 2. Create a file in the workspace

Example:

`Erstelle einen Unterordner demo und darin notes.txt`

Expected behavior:

- manager routes to coding
- coding plan contains:
  - `make_directory demo`
  - `create_file demo/notes.txt`
- file is created in the thread workspace

### 3. Create Python file + approval + execution

Example:

`Bitte erstelle eine kleine Python-Datei, die Hello World ausgibt, und beantrage anschließend die Ausführung.`

Expected behavior:

- coding creates a structured action batch
- typically:
  - `make_directory src`
  - `create_file src/main.py`
  - `request_execution src/main.py`
- manager creates an approval request
- push notification is sent
- after approval, Python is executed via `bubblewrap` when possible

### 4. Blocked sensitive path access

Example:

`Bitte lies .env`

Expected behavior:

- request is blocked
- no file read happens
- no approval is created
- logs show the blocked path and reason

## Current Limitations

AI Hub is useful, but not complete. Current limitations include:

- The research agent is still intentionally simple and mostly analytical text output.
- The manager can now produce structured coding plans, but fallback generation is still partly heuristic.
- The coding agent still has a heuristic parsing fallback for cases where no structured plan is provided.
- The system is optimized for controlled local use, not general-purpose autonomous orchestration.
- There is no full production deployment/ops layer yet.
- Tailscale exposure is assumed to be host-specific and manually maintained.
- The subprocess execution fallback is less isolated than `bubblewrap`, even though it is clearly surfaced and logged.
- The manager uses Ollama for planning, but the quality of plans still depends on model quality and prompt adherence.

## Next Sensible Steps / Roadmap

The next development steps that make sense from the current state are:

### 1. Manager polish

- improve manager answer quality
- reduce awkward or repetitive phrasing
- keep strategic and operational responses well separated

### 2. Research agent expansion

- move from placeholder analysis text to structured read-only research flows
- define clearer source handling and result structure

### 3. Stronger structured delegation

- continue reducing heuristic parsing
- let the manager produce more reliable structured coding plans
- validate multi-step plans more strictly

### 4. Operational convenience

- add startup helpers
- add systemd/service management
- add local deployment/runbook guidance

### 5. Documentation and onboarding

- keep this README current
- add architecture diagrams
- add troubleshooting playbooks
- add environment setup examples

### 6. Safer execution hardening

- keep improving the execution sandbox path
- reduce reliance on subprocess fallback
- add more execution-policy tests

## Quick Orientation for a New Codex Chat

If a new Codex chat needs to understand AI Hub quickly, the best reading order is:

1. this README
2. [workflow.py](src/ai_hub/orchestration/workflow.py)
3. [coding_agent.py](src/ai_hub/agents/coding_agent.py)
4. [coding_actions.py](src/ai_hub/schemas/coding_actions.py)
5. [coding_delegation.py](src/ai_hub/schemas/coding_delegation.py)
6. [file_tools.py](src/ai_hub/tools/file_tools.py)
7. [code_runner.py](src/ai_hub/tools/code_runner.py)
8. [app.py](src/ai_hub/web/app.py)
9. [docs/logging.md](docs/logging.md)

That path gives the fastest correct understanding of:

- what the system does
- how messages move through it
- where safety is enforced
- where structured delegation now exists

## Summary

AI Hub is a private local manager-led multi-agent system with:

- a usable web app
- persistent chat threads
- an Ollama-backed manager
- structured coding delegation
- approval-gated Python execution
- workspace enforcement
- mobile access and push notifications
- traceable logs

It is already operational enough for real iterative use, while still intentionally conservative in autonomy and security.
