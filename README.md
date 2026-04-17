# AI Hub

AI Hub is a private local multi-agent system that runs on a personal machine and is exposed in a controlled way through a FastAPI web app. The project is designed around a manager-driven workflow: the user talks to a manager, the manager plans and routes the request, and specialized workers handle bounded tasks inside explicit security limits.

This repository is both the application and the operating handbook for the current system state. It is meant to be the first document a new Codex chat, the project owner, or a technical reviewer should read.

## Project Overview

### What AI Hub is

AI Hub is a local/private orchestration platform for:

- a web-based chat interface
- persistent thread-based conversations
- a manager agent backed by Ollama
- specialized coding, research, and review agents backed by Ollama
- bounded worker delegation
- step-based project planning and autonomous multi-step execution loops
- approval-gated Python execution
- approval-gated dependency installation inside per-workspace virtualenvs
- workspace-restricted coding tasks
- push notifications for approval events

The system is intentionally conservative. It is not trying to be a fully autonomous agent runtime. The current architecture prioritizes:

- traceability
- human approval before execution
- bounded autonomy instead of open-ended autonomy
- strict workspace boundaries
- incremental delegation instead of uncontrolled autonomy
- reusable thread-level project memory instead of raw full-history prompting

### Vision

The long-term vision is a usable private multi-agent assistant where:

- the user primarily interacts with a manager agent
- the manager maintains context across threads
- coding, research, and review workers operate in specialized roles
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
- manager workflow with Ollama and visible error reporting
- coding delegation with structured internal actions
- research delegation with LLM-backed analytical output
- review delegation with LLM-backed critical second-opinion output
- thread-scoped project artifacts for reusable internal memory
- manager-side project planning route for roadmap-first collaboration
- manager-side pre-coding consultation via research and review for larger coding requests
- autonomous manager loop over stored project steps with resume-from-state behavior
- coding self-check summaries after each action batch
- post-coding implementation review by the reviewer agent when the slice is substantive enough
- dependency audits against workspace imports, `requirements.txt`, and workspace `.venv`
- provider-backed web search for the research agent with optional page previews
- approval requests for Python execution
- approval requests for missing Python package installation
- execution after user approval
- debug-oriented reruns using the latest execution error context
- workspace-restricted file operations
- sensitive path blocking
- push notifications
- logging/tracing across manager, execution, and API flows

## Core Capabilities

### User-facing capabilities

- Private web app with login
- Thread-based chat UX
- Immediate local message rendering after send
- Persistent Thinking indicator with live elapsed time in the thread
- Short live status comment below the Thinking indicator
- Mobile usage via iPhone home-screen web app
- Tailscale-based private remote access
- Push notifications when an approval is needed

### Manager capabilities

- Natural user interaction
- Thread-aware context handling
- Compact project-memory reuse across turns
- Routing between direct answer, project planning, coding delegation, research delegation, and review delegation
- Ollama-backed planning with explicit in-chat error feedback
- Project-plan synthesis with repo structure, implementation steps, validation steps, and completion criteria
- Project-state tracking across turns so follow-up feedback can resume the current implementation state instead of restarting
- Autonomous multi-step execution with stop conditions such as `ready_to_test`, approval-required, and blocked/error
- Internal research + review consultation before larger coding tasks
- External/web research via the research agent when configured
- Aggregated progress and completion summaries built from the full autonomous run
- Approval coordination
- Logging of routing, delegation decisions, and LLM failures

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
- Internal self-check summaries for touched files and obvious follow-up risks
- Normalization of test-file execution requests to module-based test runs
- Approval request creation only after the required file exists
- LLM-driven action planning without silent heuristic fallback

### Research capabilities

- Read-only analytical research responses
- Optional provider-backed web search
- Optional page-preview fetching for top web results
- Source metadata returned in internal research payloads
- Reusable research artifacts stored per thread

### Execution capabilities

- Python execution only after approval
- per-workspace `.venv` handling for approved runtime and dependency actions
- pip bootstrap inside workspace virtualenvs when needed
- `bubblewrap` sandbox as preferred execution backend
- controlled fallback to subprocess mode when `bubblewrap` cannot be used
- workspace-only execution target validation

## Architecture

### High-level architecture

`User -> Web App/API -> Manager Workflow -> Planning + Project Memory -> Optional Research/Review Consultation -> Worker Delegation -> Tools -> Approval/Execution`

### Main components

#### User

The user interacts through the web app. The primary interface is a chat thread. The user can also approve or reject Python execution requests.

#### Web app / API

The web layer lives in [app.py](src/ai_hub/web/app.py) and serves:

- login/session handling
- static web UI
- thread APIs
- asynchronous chat enqueue API
- approval APIs
- push subscription APIs

The frontend lives in [index.html](src/ai_hub/web/static/index.html) and the service worker in [sw.js](src/ai_hub/web/static/sw.js).

#### Manager

The manager runtime is implemented in [workflow.py](src/ai_hub/orchestration/workflow.py). It is the central orchestrator and is responsible for:

- storing user messages
- preparing manager planning input
- enriching that planning input with compact thread-scoped project memory
- calling the Ollama-backed planner
- validating and applying the structured planner result
- consulting research and review workers before larger coding tasks
- delegating to workers
- creating approval requests
- surfacing LLM and orchestration errors back into the chat thread
- completing pending assistant messages that were enqueued by the web API
- sending push notifications

The manager prompt is in [manager.txt](src/ai_hub/prompts/manager.txt).

#### Coding agent

The coding worker is implemented in [coding_agent.py](src/ai_hub/agents/coding_agent.py). It runs on structured internal actions and uses its dedicated model to plan action batches.

Its execution pipeline is:

1. receive a structured delegation plan if available
2. otherwise request a structured action batch from the coding model
3. inherit manager-prepared decomposition context for larger tasks when available
4. execute actions in order
5. build a compact self-check summary from the changed files and pending follow-up risks
6. return structured execution results
7. optionally prepare an approval request

The coding prompt is in [coding_agent.txt](src/ai_hub/prompts/coding_agent.txt).

#### Research agent

The research worker is implemented in [research_agent.py](src/ai_hub/agents/research_agent.py). It uses its dedicated model to produce structured analytical output for comparisons, summaries, and open questions. It is read-only and does not touch the coding workspace.
When configured, it can enrich its prompt with provider-backed web search results and fetched page previews before synthesizing an answer.

#### Reviewer agent

The review worker is implemented in [reviewer_agent.py](src/ai_hub/agents/reviewer_agent.py). It acts as a critical second opinion for plans, implementation ideas, and results. It is read-only and focuses on risks, gaps, regressions, and missing checks.

#### Structured schemas

Important internal schemas live in [src/ai_hub/schemas](src/ai_hub/schemas):

- [coding_actions.py](src/ai_hub/schemas/coding_actions.py)
- [coding_delegation.py](src/ai_hub/schemas/coding_delegation.py)
- [manager_plan.py](src/ai_hub/schemas/manager_plan.py)

These schemas are important because they define the structured contract between manager planning and coding execution.

#### Thread artifacts / project memory

In addition to raw chat messages, AI Hub now stores compact thread-scoped artifacts in SQLite so that key internal state can be reused without replaying the whole conversation.

Current artifact categories include:

- `project_brief`
- `project_plan`
- `project_state`
- `research_notes`
- `web_research_notes`
- `review_notes`
- `coding_status`
- `implementation_review`

These artifacts are reused by the manager as compact planning context and are also exposed through the thread API.

#### Workspace tools

Low-level coding tools live in [file_tools.py](src/ai_hub/tools/file_tools.py). They enforce:

- per-thread workspace roots
- relative-path-only access
- path escape blocking
- sensitive path blocking

#### Web research tools

Web-search integration lives in [web_search.py](src/ai_hub/tools/web_search.py). The current implementation supports provider-backed search with:

- `Brave Search API`
- `Tavily Search API`

The provider is selected through configuration. When enabled, the research agent can search the web, attach compact source metadata, and fetch short page previews for the top results.

#### Approval and execution

Execution policy is implemented in [code_runner.py](src/ai_hub/tools/code_runner.py). This module validates Python execution requests, chooses the sandbox backend, and performs the actual execution only after approval.

#### Push notifications

Push behavior is implemented in [push_notify.py](src/ai_hub/tools/push_notify.py) and web push subscriptions are stored in SQLite.

#### Ollama integration

Ollama integration lives in:

- [manager_planner.py](src/ai_hub/llm/manager_planner.py)
- [ollama_client.py](src/ai_hub/llm/ollama_client.py)
- [model_router.py](src/ai_hub/llm/model_router.py)

The currently configured role-to-model mapping is:

- manager: `gemma4:31b`
- coding: `qwen3-coder:30b`
- research: `qwen3:30b`
- reviewer: `qwen3-coder:30b`

The backend still enforces safety, workspace limits, and approval rules independently of any model output.
Web research remains provider-gated and configuration-driven; it is not silently enabled without a configured search backend.

#### Logging

Central logging configuration is in [logging_config.py](src/ai_hub/logging_config.py). Logs are written to the `logs/` directory with rotating file handlers.

## Message / Request Flow

### Step-by-step flow

1. The user sends a chat message through the web app.
2. The frontend shows the user message immediately in the chat log.
3. `POST /api/chat` in [app.py](src/ai_hub/web/app.py) creates or reuses the thread, stores the user message, and writes a pending assistant message.
4. The frontend renders that pending assistant message as a Thinking state with animated dots and a live timer.
5. The frontend shows a short live status comment under Thinking, for example which phase or worker is currently active.
6. A background worker in the backend processes the queued message through the manager workflow.
7. The manager planner asks Ollama for a structured plan.
8. The workflow validates the structured manager plan from Ollama.
9. For strategic or roadmap-style requests, the manager may switch into a dedicated planning route before coding starts.
10. In that planning route, the manager consults the research and reviewer workers internally and comes back with a project proposal plus feedback questions.
11. For larger coding requests, the manager may still consult the research and reviewer workers internally before choosing the next bounded implementation step.
12. If autonomous project execution is enabled, the manager can continue through stored project steps until it reaches a real stop condition such as `ready_to_test`, approval-required, blocked/error, or the configured safety budget.
13. After coding work, the manager may ask the reviewer agent to critique the latest implementation step using the coding summary and self-check.
14. The manager stores compact project artifacts such as project plans, project state, research notes, review notes, project briefs, coding status, dependency status, and implementation reviews.
15. The result is one of:
   - direct manager reply
   - project plan
   - coding delegation
   - research delegation
   - review delegation
16. If the route is `research`, the research agent may use configured web search and page previews when the task depends on current or external information.
17. If the route is `coding`, the manager may provide a structured `CodingDelegationPlan`.
18. The coding agent prefers that structured plan.
19. If no structured coding plan is provided, the coding agent requests a structured action batch from its own model.
20. The coding agent executes actions through the server-side tool layer.
21. After a coding step, the manager can run a dependency audit and prepare an approval request for missing package installation inside the workspace virtualenv.
22. If execution is requested, the coding agent prepares an approval request instead of running Python directly.
23. The manager stores approval requests and sends a push notification.
24. The pending assistant message is updated in place with the final manager reply or an explicit error.
25. The frontend polls the active thread while a Thinking message exists and stops polling when processing is finished.
26. Only the latest still-open processing message is rendered as an active Thinking block in the UI.
27. The user approves or rejects execution or dependency-install requests in the web app.
28. If approved, the backend validates the command again and runs Python through `bubblewrap` when possible.
29. Successful and failed results are logged.

### Project planning flow

For collaborative multi-turn project work, the manager can now run a dedicated planning loop before coding starts:

1. interpret the request as a project-planning task instead of immediate implementation
2. ask the research agent for feasibility, architecture direction, repo shape, milestones, and the safest first implementation step
3. ask the reviewer agent to challenge scope, missing validation, and places where user confirmation should be requested
4. synthesize both results into a manager-owned proposal
5. store that proposal as `project_plan` and `project_state`
6. come back to the user with a concrete recommendation and explicit feedback points

This makes the manager more suitable for long-running collaborative project delivery instead of one-shot coding only.

After a project plan is stored, short user replies such as approval, agreement, or "continue" can now resume the next bounded implementation step from that saved project state instead of restarting from scratch.

### Internal consultation flow for larger coding tasks

For coding requests that look too large for a one-shot implementation, the manager now follows a conservative internal loop:

1. ask the research agent to break the request into smaller work packages, assumptions, and risks
2. ask the reviewer agent to critique that decomposition and challenge oversized scope
3. compress both results into thread-scoped artifacts
4. pass the smallest safe next implementation step to the coding agent
5. let the coding agent produce a self-check summary after execution
6. let the reviewer agent critique the latest coding step when the step is substantive enough to justify the extra loop

This keeps the external UX manager-centric while allowing more natural internal multi-step coordination.

When autonomous execution is enabled, the manager can repeat that loop across multiple stored steps without needing a fresh user nudge after each step. The loop pauses only at real control points such as approval-required, `ready_to_test`, blocked/error, or when the configured safety budget is exhausted.

### Web research flow

When web research is enabled and the task calls for external/current information, the research path becomes:

1. derive a compact search query from the research task
2. call the configured search provider
3. fetch short previews for the top result pages
4. hand both snippets and previews to the research model
5. store source metadata and summarized findings in thread artifacts

When source metadata exists, the workflow additionally stores a dedicated `web_research_notes` artifact so later manager and reviewer steps can reuse externally grounded findings without rerunning the search immediately.

### Textual flow diagram

`User`
-> `FastAPI /api/chat`
-> `store user message + pending assistant message`
-> `background worker`
-> `ManagerWorkflow`
-> `ManagerPlanner (Ollama)`
-> `LLM plan / explicit error response`
-> `DelegationService`
-> `CodingAgent / ResearchAgent / ReviewerAgent`
-> `Workspace tools / approval system`
-> `pending message updated in place`
-> `frontend polling refresh`
-> `User`

### Approval flow

`Coding request`
-> `structured action batch`
-> `request_execution action` or dependency audit result
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

The same approval mechanism is also used for dependency installation when the workspace dependency audit detects packages that are missing from the workspace virtualenv or missing from the workspace `requirements.txt`.

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

### LLM failure behavior

If Ollama is unavailable, misconfigured, or returns unusable output:

- the failing component returns an explicit error into the chat thread
- the error includes model and failure details when available
- no silent heuristic substitution is used for manager planning or worker reasoning
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
- [agents/coding_agent.py](src/ai_hub/agents/coding_agent.py): LLM-backed structured coding action planning and execution
- [agents/research_agent.py](src/ai_hub/agents/research_agent.py): LLM-backed read-only research worker
- [agents/reviewer_agent.py](src/ai_hub/agents/reviewer_agent.py): LLM-backed read-only reviewer worker

#### Orchestration

- [orchestration/workflow.py](src/ai_hub/orchestration/workflow.py): central manager workflow
- [orchestration/delegation.py](src/ai_hub/orchestration/delegation.py): worker dispatch

#### Schemas

- [schemas/manager_plan.py](src/ai_hub/schemas/manager_plan.py): manager LLM plan model
- [schemas/coding_actions.py](src/ai_hub/schemas/coding_actions.py): internal coding actions
- [schemas/coding_delegation.py](src/ai_hub/schemas/coding_delegation.py): manager-to-coding structured handoff

#### Memory

- [memory/sqlite_db.py](src/ai_hub/memory/sqlite_db.py): SQLite persistence layer
- [memory/store.py](src/ai_hub/memory/store.py): higher-level store API including message updates
- [memory/history.py](src/ai_hub/memory/history.py): thread history formatting

#### LLM

- [llm/manager_planner.py](src/ai_hub/llm/manager_planner.py): Ollama-backed manager planner
- [llm/ollama_client.py](src/ai_hub/llm/ollama_client.py): low-level Ollama HTTP client
- [llm/model_router.py](src/ai_hub/llm/model_router.py): model mapping by role

#### Tools

- [tools/file_tools.py](src/ai_hub/tools/file_tools.py): workspace filesystem enforcement
- [tools/code_runner.py](src/ai_hub/tools/code_runner.py): execution validation and sandboxed execution
- [tools/web_search.py](src/ai_hub/tools/web_search.py): provider-backed web search and page-preview fetching
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
- [prompts/reviewer_agent.txt](src/ai_hub/prompts/reviewer_agent.txt)

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

Optional web-research settings:

- `WEB_SEARCH_ENABLED=true`
- `WEB_SEARCH_PROVIDER=auto|brave|tavily`
- `BRAVE_SEARCH_API_KEY=...`
- `TAVILY_API_KEY=...`
- `WEB_SEARCH_MAX_RESULTS=5`
- `WEB_SEARCH_FETCH_PAGES=true`
- `WEB_SEARCH_FETCH_TOP_N=2`

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
- background chat processing
- manager routing decisions
- `source=ollama|error`
- `llm_decision`
- `final_decision`
- planner and orchestration errors
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
- check `source=ollama` vs `source=error`

If coding actions are wrong:

- inspect `logs/execution.log`
- look for `coding_action_batch`
- verify `structured_plan_used` and `action_execution_order`

If approval/execution fails:

- inspect `approval_created`, `approval_approved`, `approval_execution_finished`
- check `sandbox_backend`
- inspect `returncode`, `stderr`, and blocked reasons

## Typical Workflows

### 1. Strategic question to the manager

Example:

`Wie würdest du ein kleines Python-Projekt strukturieren, das später über Tailscale und Web Push erreichbar ist?`

Expected behavior:

- manager answers directly if the planner chooses `direct`
- no coding workspace action is executed
- no approval is created

### 2. Create a file in the workspace

Example:

`Erstelle einen Unterordner demo und darin notes.txt`

Expected behavior:

- manager routes to coding
- coding plan contains structured actions such as:
  - `make_directory demo`
  - `create_file demo/notes.txt`
- file is created in the thread workspace

### 3. Create Python file + approval + execution

Example:

`Bitte erstelle eine kleine Python-Datei, die Hello World ausgibt, und beantrage anschließend die Ausführung.`

Expected behavior:

- coding creates or receives a structured action batch
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

### 5. Visible LLM error in chat

Example:

`Was kannst du aktuell in diesem System tun?`

Expected behavior when the responsible model is unavailable or returns invalid output:

- the thread still receives an assistant reply
- the reply contains an explicit error for the failing component
- the reply includes model and error details when available
- no silent fallback answer is generated

### 6. Thinking indicator during processing

Example:

`Erstelle bitte ein kleines Python-Projekt mit README und requirements.txt`

Expected behavior:

- the user message appears immediately after pressing send
- a pending manager message appears immediately as `Thinking`
- animated black dots are shown while processing is still running
- the elapsed Thinking time updates once per second
- a short grey status line explains the current phase in compact form
- if the page is reloaded while processing is still running, the thread still shows the same Thinking state and elapsed time
- when the backend finishes, the pending manager message is replaced by the final answer
- older completed Thinking blocks do not remain active in the chat UI

## Current Limitations

AI Hub is useful, but not complete. Current limitations include:

- The system currently depends on valid LLM output for manager planning and worker reasoning.
- If an LLM is unavailable or returns invalid output, the failure is surfaced directly in the chat instead of being auto-repaired.
- The research and review agents are useful, but still intentionally read-only and narrow in tool use.
- The current chat processing model uses polling from the frontend while a pending Thinking message exists.
- The system is optimized for controlled local use, not general-purpose autonomous orchestration.
- There is no full production deployment/ops layer yet.
- Tailscale exposure is assumed to be host-specific and manually maintained.
- The subprocess execution fallback is less isolated than `bubblewrap`, even though it is clearly surfaced and logged.
- The quality of manager plans and worker outputs still depends strongly on model quality, prompt adherence, and the size of the selected implementation slice.

## Next Sensible Steps / Roadmap

The next development steps that make sense from the current state are:

### 1. Manager polish

- improve manager answer quality
- reduce awkward or repetitive phrasing
- keep strategic and operational responses well separated

### 2. Research and review expansion

- add more explicit source handling for research work
- define richer review contracts for implementation results
- decide whether either read-only worker should gain bounded tool access

### 3. Stronger structured delegation

- let the manager produce more reliable structured coding plans
- validate multi-step plans more strictly
- tighten plan validation and error reporting further

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
4. [research_agent.py](src/ai_hub/agents/research_agent.py)
5. [reviewer_agent.py](src/ai_hub/agents/reviewer_agent.py)
6. [coding_actions.py](src/ai_hub/schemas/coding_actions.py)
7. [coding_delegation.py](src/ai_hub/schemas/coding_delegation.py)
8. [file_tools.py](src/ai_hub/tools/file_tools.py)
9. [code_runner.py](src/ai_hub/tools/code_runner.py)
10. [app.py](src/ai_hub/web/app.py)
11. [docs/logging.md](docs/logging.md)

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
- dedicated coding, research, and review workers
- structured coding delegation
- approval-gated Python execution
- workspace enforcement
- visible in-chat error reporting for LLM failures
- mobile access and push notifications
- traceable logs

It is already operational enough for real iterative use, while still intentionally conservative in autonomy and security.
