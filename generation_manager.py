"""后台生成任务管理，避免 Streamlit 页面切换中断 Agent 回复。"""
import queue
import threading
import time
import uuid


_jobs = {}
_jobs_lock = threading.RLock()
_queue = queue.Queue()
_worker_thread = None


def _ensure_worker():
    global _worker_thread
    with _jobs_lock:
        if _worker_thread and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, name="rag-agent-worker", daemon=True)
        _worker_thread.start()


def submit_prompt(agent_service, session_id, prompt):
    """保存用户输入和 AI 占位消息，并把生成任务放入后台队列"""
    job_id = uuid.uuid4().hex[:12]
    agent_service.add_user_message_with_ai_placeholder(session_id, prompt, job_id)
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "session_id": session_id,
            "prompt": prompt,
            "agent_service": agent_service,
            "status": "queued",
            "response": "",
            "error": "",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    _queue.put(job_id)
    _ensure_worker()
    return job_id


def cancel_session_jobs(session_id):
    """取消指定会话中尚未完成的后台任务"""
    with _jobs_lock:
        for job in _jobs.values():
            if job["session_id"] == session_id and job["status"] in {"queued", "running"}:
                job["status"] = "cancelled"
                job["updated_at"] = time.time()


def has_active_jobs(session_id=None):
    with _jobs_lock:
        for job in _jobs.values():
            if session_id is not None and job["session_id"] != session_id:
                continue
            if job["status"] in {"queued", "running"}:
                return True
    return False


def get_session_jobs(session_id, include_finished=False):
    with _jobs_lock:
        jobs = [job.copy() for job in _jobs.values() if job["session_id"] == session_id]
    if not include_finished:
        jobs = [job for job in jobs if job["status"] in {"queued", "running"}]
    return sorted(jobs, key=lambda job: job["created_at"])


def _is_cancelled(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return not job or job["status"] == "cancelled"


def _set_job_status(job_id, status, response=None, error=None):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        if job["status"] == "cancelled":
            return job.copy()
        job["status"] = status
        if response is not None:
            job["response"] = response
        if error is not None:
            job["error"] = error
        job["updated_at"] = time.time()
        return job.copy()


def _worker_loop():
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        finally:
            _queue.task_done()


def _run_job(job_id):
    job = _set_job_status(job_id, "running", response="")
    if not job or _is_cancelled(job_id):
        return

    agent_service = job["agent_service"]
    session_id = job["session_id"]
    prompt = job["prompt"]
    response = ""
    last_write_at = 0.0
    agent_service.update_ai_message(session_id, job_id, content="", status="running")

    try:
        for chunk in agent_service.stream(prompt, session_id, persist_user=False, persist_ai=False):
            if _is_cancelled(job_id):
                return
            response += chunk
            now = time.monotonic()
            if now - last_write_at >= 0.25:
                _set_job_status(job_id, "running", response=response)
                agent_service.update_ai_message(session_id, job_id, content=response, status="running")
                last_write_at = now

        if _is_cancelled(job_id):
            return

        _set_job_status(job_id, "done", response=response)
        agent_service.update_ai_message(session_id, job_id, content=response, status="done")
    except Exception as exc:
        message = f"回复生成失败：{exc}"
        _set_job_status(job_id, "error", response=message, error=str(exc))
        if not _is_cancelled(job_id):
            agent_service.update_ai_message(session_id, job_id, content=message, status="error", error=str(exc))
