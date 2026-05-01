import os
import json
import threading
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict
from rag_project import config


_LOCKS = {}
_LOCKS_LOCK = threading.Lock()


def _get_lock(file_path):
    with _LOCKS_LOCK:
        if file_path not in _LOCKS:
            _LOCKS[file_path] = threading.RLock()
        return _LOCKS[file_path]


def _normalize_session_id(session_id):
    session_id = str(session_id).strip()
    if not session_id or os.path.basename(session_id) != session_id:
        raise ValueError("非法 session_id")
    return session_id


def get_history(session_id):
    return FileChatMessageHistory(session_id, config.chat_history_dir)


def list_history_sessions(storage_path=None):
    storage_path = storage_path or config.chat_history_dir
    if not os.path.exists(storage_path):
        return []

    sessions = []
    for name in os.listdir(storage_path):
        path = os.path.join(storage_path, name)
        if os.path.isfile(path) and os.path.getsize(path) > 2:
            sessions.append((name, os.path.getmtime(path)))
    return [name for name, _ in sorted(sessions, key=lambda item: item[1], reverse=True)]


def delete_history(session_id, storage_path=None):
    storage_path = storage_path or config.chat_history_dir
    session_id = _normalize_session_id(session_id)
    file_path = os.path.join(storage_path, session_id)
    if os.path.exists(file_path):
        os.remove(file_path)


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path=None):
        self.session_id = _normalize_session_id(session_id)
        self.storage_path = storage_path or config.chat_history_dir
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(self.storage_path, exist_ok=True)
        self._lock = _get_lock(self.file_path)

    def _read_message_dicts(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_message_dicts(self, messages_data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_data, f, ensure_ascii=False)

    def add_messages(self, messages):
        with self._lock:
            messages_data = self._read_message_dicts()
            messages_data.extend(message_to_dict(message) for message in messages)
            self._write_message_dicts(messages_data)

    def update_ai_message(self, job_id, content=None, status=None, error=None):
        """按 job_id 原位更新 AI 占位消息，保留它在对话中的顺序"""
        with self._lock:
            messages_data = self._read_message_dicts()
            for message in reversed(messages_data):
                data = message.get("data", {})
                kwargs = data.get("additional_kwargs") or {}
                if message.get("type") == "ai" and kwargs.get("job_id") == job_id:
                    if content is not None:
                        data["content"] = content
                    if status is not None:
                        kwargs["status"] = status
                    if error is not None:
                        kwargs["error"] = error
                    data["additional_kwargs"] = kwargs
                    self._write_message_dicts(messages_data)
                    return True
        return False

    @property
    def messages(self):
        with self._lock:
            return messages_from_dict(self._read_message_dicts())

    def clear(self):
        with self._lock:
            self._write_message_dicts([])
