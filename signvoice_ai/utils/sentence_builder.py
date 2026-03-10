"""
Sentence builder for gesture streams.

Converts a stream of recognized gestures into a readable sentence.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional


class SentenceBuilder:
    """Builds a sentence from recognized gestures with basic controls."""

    _PUNCTUATION = {".", ",", "?", "!", ":", ";"}

    _CONTROL_ACTIONS = {
        "SPACE": "space",
        "ПРОБЕЛ": "space",
        "BACKSPACE": "backspace",
        "DELETE": "backspace",
        "УДАЛИТЬ": "backspace",
        "CLEAR": "clear",
        "СБРОС": "clear",
        "ОЧИСТИТЬ": "clear",
        "DOT": "punct",
        "PERIOD": "punct",
        "ТОЧКА": "punct",
        "COMMA": "comma",
        "ЗАПЯТАЯ": "comma",
        "QUESTION": "question",
        "ВОПРОС": "question",
        "QUESTION_MARK": "question",
        "EXCLAMATION": "exclamation",
        "ВОСКЛИЦ": "exclamation",
    }

    def __init__(self, max_tokens: int = 40, dedupe_window_s: float = 0.7):
        self.max_tokens = max_tokens
        self.dedupe_window_s = dedupe_window_s
        self.tokens: List[str] = []
        self._last_added: Optional[str] = None
        self._last_added_ts: float = 0.0

    def reset(self) -> None:
        self.tokens = []
        self._last_added = None
        self._last_added_ts = 0.0

    def add_gesture(self, gesture: str) -> bool:
        if not gesture:
            return False

        now = time.monotonic()
        if gesture == self._last_added and (now - self._last_added_ts) < self.dedupe_window_s:
            return False

        self._last_added = gesture
        self._last_added_ts = now

        action = self._action_for(gesture)
        if action == "clear":
            self.reset()
            return True
        if action == "backspace":
            self._backspace()
            return True
        if action == "space":
            self._ensure_space_slot()
            return True
        if action in {"punct", "comma", "question", "exclamation"}:
            punct = self._punct_for_action(action)
            self._append_punctuation(punct)
            return True

        if self._is_letter_token(gesture):
            self._append_letter(gesture)
        else:
            self._append_word(gesture)

        self._trim()
        return True

    def get_sentence(self) -> str:
        raw = " ".join(token for token in self.tokens if token != "")
        normalized = re.sub(r"\s+([,.!?:;])", r"\1", raw)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        return normalized

    def _action_for(self, gesture: str) -> Optional[str]:
        normalized = gesture.strip().upper()
        for key, action in self._CONTROL_ACTIONS.items():
            if normalized == key or normalized.startswith(key):
                return action
        return None

    def _punct_for_action(self, action: str) -> str:
        if action == "comma":
            return ","
        if action == "question":
            return "?"
        if action == "exclamation":
            return "!"
        return "."

    def _append_punctuation(self, punct: str) -> None:
        if not self.tokens:
            return
        if self.tokens[-1] == "":
            self.tokens.pop()
        if not self.tokens:
            return
        if self.tokens[-1][-1:] in self._PUNCTUATION:
            return
        self.tokens[-1] += punct

    def _ensure_space_slot(self) -> None:
        if not self.tokens or self.tokens[-1] != "":
            self.tokens.append("")

    def _append_letter(self, gesture: str) -> None:
        letter = self._normalize_letter(gesture)
        if not self.tokens:
            self.tokens.append(letter)
            return
        if self.tokens[-1] == "":
            self.tokens[-1] = letter
            return
        self.tokens[-1] += letter

    def _append_word(self, gesture: str) -> None:
        if self.tokens and self.tokens[-1] == "":
            self.tokens[-1] = gesture
        else:
            self.tokens.append(gesture)

    def _backspace(self) -> None:
        if not self.tokens:
            return
        last = self.tokens[-1]
        if last == "":
            self.tokens.pop()
            return
        if len(last) > 1:
            self.tokens[-1] = last[:-1]
        else:
            self.tokens.pop()

    def _trim(self) -> None:
        if self.max_tokens <= 0:
            return
        if len(self.tokens) > self.max_tokens:
            self.tokens = self.tokens[-self.max_tokens :]

    def _is_letter_token(self, gesture: str) -> bool:
        token = gesture.strip()
        if len(token) == 1 and token.isalpha():
            return True
        upper = token.upper()
        return upper.startswith("LETTER_") or upper.startswith("БУКВА_")

    def _normalize_letter(self, gesture: str) -> str:
        token = gesture.strip()
        upper = token.upper()
        if upper.startswith("LETTER_"):
            return token.split("_", 1)[-1]
        if upper.startswith("БУКВА_"):
            return token.split("_", 1)[-1]
        return token
