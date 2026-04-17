from pathlib import Path
import logging
import secrets
import threading
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from ai_hub.config import (
    APP_USERNAME,
    APP_PASSWORD,
    CHAT_BACKGROUND_TIMEOUT_SECONDS,
    CHAT_THREAD_DUMP_ENABLED,
    CHAT_THREAD_DUMP_INTERVAL_SECONDS,
    CHAT_THREAD_DUMP_MAX_CHARS,
    SESSION_SECRET,
    VAPID_PUBLIC_KEY,
)
from ai_hub.logging_config import (
    create_chat_run_log_dir,
    log_event,
    log_thread_dump,
    reset_chat_log_dir,
    reset_request_id,
    set_chat_log_dir,
    set_request_id,
    setup_logging,
)
from ai_hub.memory.store import HubStore
from ai_hub.memory.sqlite_db import init_db, upsert_push_subscription, delete_push_subscription
from ai_hub.orchestration.workflow import ManagerWorkflow
from ai_hub.state import ApprovalStatus
from ai_hub.tools.code_runner import ExecutionPolicyError
from ai_hub.tools.file_tools import WorkspaceSecurityError


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
logger = logging.getLogger(__name__)
setup_logging()

app = FastAPI(title="AI Hub")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET or "dev-secret-change-me",
    same_site="lax",
    https_only=False,  # später bei Tailscale HTTPS auf True setzen
)


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PushKeys(BaseModel):
    auth: str
    p256dh: str


class PushSubscriptionIn(BaseModel):
    endpoint: str
    expirationTime: int | None = None
    keys: PushKeys


class PushSubscribeRequest(BaseModel):
    subscription: PushSubscriptionIn


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


class ThreadCreateRequest(BaseModel):
    title: str | None = None


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
init_db()
store = HubStore()
workflow = ManagerWorkflow(store=store)


def _run_chat_in_background(thread_id: str, user_message: str, pending_message_id: int, request_id: str) -> None:
    request_token = set_request_id(request_id)
    run_dir = create_chat_run_log_dir(thread_id, request_id)
    chat_log_token = set_chat_log_dir(str(run_dir))
    try:
        log_event(
            logger,
            "api_chat_run_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            run_log_dir=str(run_dir),
        )
        result_holder: dict[str, object] = {}
        error_holder: dict[str, Exception] = {}

        def runner() -> None:
            nested_request_token = set_request_id(request_id)
            nested_chat_log_token = set_chat_log_dir(str(run_dir))
            try:
                log_event(
                    logger,
                    "api_chat_worker_started",
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                )
                result_holder["result"] = workflow.process_enqueued_chat(
                    thread_id=thread_id,
                    user_message=user_message,
                    pending_message_id=pending_message_id,
                )
                log_event(
                    logger,
                    "api_chat_worker_completed",
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                    result_type=type(result_holder.get("result")).__name__,
                )
            except Exception as exc:
                error_holder["error"] = exc
                log_event(
                    logger,
                    "api_chat_worker_failed",
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
            finally:
                reset_chat_log_dir(nested_chat_log_token)
                reset_request_id(nested_request_token)

        worker_thread = threading.Thread(
            target=runner,
            name=f"chat-worker-{request_id}",
            daemon=True,
        )
        worker_thread.start()
        log_event(
            logger,
            "api_chat_worker_join_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            timeout_seconds=CHAT_BACKGROUND_TIMEOUT_SECONDS,
        )
        remaining_timeout = CHAT_BACKGROUND_TIMEOUT_SECONDS
        dump_interval = max(CHAT_THREAD_DUMP_INTERVAL_SECONDS, 1.0)
        while worker_thread.is_alive() and remaining_timeout > 0:
            wait_seconds = min(dump_interval, remaining_timeout)
            worker_thread.join(timeout=wait_seconds)
            remaining_timeout -= wait_seconds
            if worker_thread.is_alive() and CHAT_THREAD_DUMP_ENABLED:
                log_event(
                    logger,
                    "api_chat_worker_still_running",
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                    remaining_timeout_seconds=round(max(remaining_timeout, 0.0), 3),
                )
                log_thread_dump(
                    logger,
                    "api_chat_thread_dump",
                    max_chars=CHAT_THREAD_DUMP_MAX_CHARS,
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                    remaining_timeout_seconds=round(max(remaining_timeout, 0.0), 3),
                )
        log_event(
            logger,
            "api_chat_worker_join_returned",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            worker_alive=worker_thread.is_alive(),
            has_error="error" in error_holder,
            has_result="result" in result_holder,
        )
        if worker_thread.is_alive():
            log_event(
                logger,
                "api_chat_background_timeout",
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                timeout_seconds=CHAT_BACKGROUND_TIMEOUT_SECONDS,
            )
            workflow.finalize_pending_timeout(
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                user_message=user_message,
                timeout_seconds=CHAT_BACKGROUND_TIMEOUT_SECONDS,
                worker_label="Kritiker-Agent / Reviewer agent",
            )
            log_event(
                logger,
                "api_chat_background_timeout_finalized",
                thread_id=thread_id,
                pending_message_id=pending_message_id,
            )
            return
        if "error" in error_holder:
            raise error_holder["error"]
    except Exception as exc:
        log_event(
            logger,
            "api_chat_background_failed",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            error=str(exc),
            exception_type=type(exc).__name__,
        )
    finally:
        reset_chat_log_dir(chat_log_token)
        reset_request_id(request_token)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
    token = set_request_id(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            log_event(
                logger,
                "api_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )
        return response
    finally:
        reset_request_id(token)

@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get("authenticated", False))


@app.get("/")
def read_index(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
def read_login(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/me")
def me(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    return {
        "authenticated": True,
        "username": request.session.get("username", ""),
    }


@app.get("/api/push/status")
def push_status(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    status = store.system_status()
    return {"authenticated": True, "subscriptions_count": status["push_subscriptions"]}


@app.get("/api/push/public-key")
def push_public_key(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    if not VAPID_PUBLIC_KEY:
        return JSONResponse(
            {"authenticated": True, "configured": False},
            status_code=500,
        )

    return {
        "authenticated": True,
        "configured": True,
        "public_key": VAPID_PUBLIC_KEY,
    }

@app.post("/api/push/subscribe")
def push_subscribe(payload: PushSubscribeRequest, request: Request):
    if not is_authenticated(request):
        return JSONResponse(
            {"success": False, "message": "Nicht eingeloggt."},
            status_code=401,
        )

    subscription_dict = payload.subscription.model_dump()
    upsert_push_subscription(subscription_dict)

    return {"success": True}


@app.post("/api/push/unsubscribe")
def push_unsubscribe(payload: PushUnsubscribeRequest, request: Request):
    if not is_authenticated(request):
        return JSONResponse(
            {"success": False, "message": "Nicht eingeloggt."},
            status_code=401,
        )

    deleted = delete_push_subscription(payload.endpoint)

    return {
        "success": True,
        "deleted": deleted,
    }


@app.post("/api/login")
def login(payload: LoginRequest, request: Request):
    username_ok = secrets.compare_digest(
        payload.username.encode("utf-8"),
        APP_USERNAME.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        payload.password.encode("utf-8"),
        APP_PASSWORD.encode("utf-8"),
    )

    if not (username_ok and password_ok):
        return JSONResponse(
            {"success": False, "message": "Ungültiger Benutzername oder Passwort."},
            status_code=401,
        )

    request.session["authenticated"] = True
    request.session["username"] = payload.username

    return {"success": True}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/status")
def system_status(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    return {
        "authenticated": True,
        "status": store.system_status(),
    }


@app.get("/api/threads")
def get_threads(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    threads = store.list_threads(include_archived=True)
    log_event(logger, "api_threads_listed", count=len(threads))
    return {"threads": threads}


@app.post("/api/threads")
def create_thread(payload: ThreadCreateRequest, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    thread = store.ensure_thread(None, title=payload.title or "Neuer Chat")
    log_event(logger, "api_thread_created", thread_id=thread["id"], title=thread["title"])
    return {"thread": thread, "messages": [], "artifacts": []}


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    thread = store.get_thread(thread_id)
    if thread is None:
        log_event(logger, "api_thread_missing", thread_id=thread_id)
        return JSONResponse({"message": "Thread nicht gefunden."}, status_code=404)

    log_event(logger, "api_thread_loaded", thread_id=thread_id)
    return {
        "thread": thread,
        "messages": store.list_messages(thread_id),
        "approvals": store.list_approvals(thread_id=thread_id),
        "artifacts": store.list_artifacts(thread_id),
    }


@app.post("/api/threads/{thread_id}/archive")
def archive_thread(thread_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    thread = store.archive_thread(thread_id)
    if thread is None:
        log_event(logger, "api_thread_archive_missing", thread_id=thread_id)
        return JSONResponse({"message": "Thread nicht gefunden."}, status_code=404)
    log_event(logger, "api_thread_archived", thread_id=thread_id)
    return {"thread": thread}


@app.post("/api/threads/{thread_id}/reset")
def reset_thread(thread_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    thread = store.reset_thread(thread_id)
    if thread is None:
        log_event(logger, "api_thread_reset_missing", thread_id=thread_id)
        return JSONResponse({"message": "Thread nicht gefunden."}, status_code=404)
    log_event(logger, "api_thread_reset", thread_id=thread_id)
    return {
        "thread": thread,
        "messages": [],
        "approvals": store.list_approvals(thread_id=thread_id),
        "artifacts": store.list_artifacts(thread_id),
    }


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    deleted = store.delete_thread(thread_id)
    if not deleted:
        log_event(logger, "api_thread_delete_missing", thread_id=thread_id)
        return JSONResponse({"message": "Thread nicht gefunden."}, status_code=404)
    log_event(logger, "api_thread_deleted", thread_id=thread_id)
    return {"deleted": True, "thread_id": thread_id}


@app.get("/api/approvals")
def list_approvals(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    approvals = store.list_approvals()
    pending = store.list_approvals(status=ApprovalStatus.PENDING.value)
    log_event(logger, "api_approvals_listed", approvals=len(approvals), pending=len(pending))
    return {
        "approvals": approvals,
        "pending": pending,
    }


@app.post("/api/approvals/{approval_id}/approve")
def approve_approval(approval_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    try:
        approval = workflow.approve_execution(approval_id)
    except KeyError:
        log_event(logger, "api_approval_approve_missing", approval_id=approval_id)
        return JSONResponse({"message": "Freigabe nicht gefunden."}, status_code=404)

    log_event(logger, "api_approval_approved", approval_id=approval_id, thread_id=approval["thread_id"], status=approval["status"])
    return {"approval": approval}


@app.post("/api/approvals/{approval_id}/reject")
def reject_approval(approval_id: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"authenticated": False}, status_code=401)

    try:
        approval = workflow.reject_execution(approval_id)
    except KeyError:
        log_event(logger, "api_approval_reject_missing", approval_id=approval_id)
        return JSONResponse({"message": "Freigabe nicht gefunden."}, status_code=404)

    log_event(logger, "api_approval_rejected", approval_id=approval_id, thread_id=approval["thread_id"], status=approval["status"])
    return {"approval": approval}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    if not is_authenticated(request):
        return JSONResponse(
            {"reply": "Nicht eingeloggt."},
            status_code=401,
        )

    user_message = payload.message.strip()

    if not user_message:
        log_event(logger, "api_chat_empty", thread_id=payload.thread_id)
        return {"reply": "Bitte sende eine nicht-leere Nachricht."}

    try:
        thread = store.ensure_thread(payload.thread_id)
        user_entry = store.add_message(thread["id"], role="user", content=user_message)
        store.rename_thread_from_first_message(thread["id"], user_message)
        pending_message = store.add_message(
            thread["id"],
            role="assistant",
            content="Thinking",
            agent="manager",
            meta={
                "route": "pending",
                "manager_source": "pending",
                "final_decision": "pending",
                "processing": True,
                "processing_started_at": user_entry["created_at"],
                "thinking_label": "Thinking",
            },
        )
        background_thread = threading.Thread(
            target=_run_chat_in_background,
            args=(thread["id"], user_message, pending_message["id"], getattr(request.state, "request_id", uuid4().hex[:12])),
            daemon=True,
        )
        background_thread.start()
        result = {
            "thread": store.get_thread(thread["id"]),
            "message": pending_message,
            "reply": pending_message["content"],
            "route": "pending",
            "approval_request": None,
            "messages": store.list_messages(thread["id"]),
        }
    except (WorkspaceSecurityError, ExecutionPolicyError) as exc:
        log_event(logger, "api_chat_blocked", thread_id=payload.thread_id, error_code=getattr(exc, "code", "request_blocked"))
        return JSONResponse(
            {
                "message": getattr(exc, "user_message", str(exc)),
                "error_code": getattr(exc, "code", "request_blocked"),
            },
            status_code=400,
        )
    except ValueError as exc:
        log_event(logger, "api_chat_invalid", thread_id=payload.thread_id, error=str(exc))
        return JSONResponse({"message": str(exc)}, status_code=400)
    log_event(
        logger,
        "api_chat_queued",
        thread_id=result["thread"]["id"],
        route=result["route"],
        approval_created=False,
    )
    return result
