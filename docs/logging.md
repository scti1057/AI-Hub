# AI Hub Logging

AI Hub writes rotating UTF-8 log files to `logs/` in the project root.

Files:
- `logs/app.log`: all application events across web, manager, workspace, execution, Ollama, and push.
- `logs/manager.log`: manager planning and routing only, including `source`, `llm_decision`, `final_decision`, and approval creation.
- `logs/execution.log`: coding agent actions, workspace tools, execution backend usage, blocked paths, and push delivery events.

Useful events:
- `chat_received`, `manager_decision`, `approval_created`, `approval_approved`, `approval_rejected`
- `coding_action_plan`, `make_directory`, `write_file`, `read_file`, `delete_path`, `blocked_path`
- `execution_request_built`, `execution_request_validated`, `execution_finished`
- `api_chat_completed`, `api_thread_created`, `api_approvals_listed`
- `push_sent`, `push_failed`

Every line includes:
- timestamp
- level
- logger name
- `request_id`
- structured `event=... key=value ...` fields

Quick commands:

```bash
tail -n 80 logs/app.log
tail -n 80 logs/manager.log
tail -n 80 logs/execution.log
tail -f logs/app.log
```

Relevant environment variables:
- `LOG_LEVEL`
- `LOG_FILE_MAX_BYTES`
- `LOG_FILE_BACKUP_COUNT`
