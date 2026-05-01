import os
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict
import config_data as config


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

    def add_messages(self, messages):
        all_messages = list(self.messages)
        all_messages.extend(messages)
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False)

    @property
    def messages(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                messages_data = json.loads(content)
            return messages_from_dict(messages_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
