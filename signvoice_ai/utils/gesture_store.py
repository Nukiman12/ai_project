from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class GestureStore:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        if path is None:
            self.path = Path(__file__).resolve().parents[1] / "data_dynamic" / "saved_gestures.json"
        else:
            self.path = Path(path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_records_unsafe(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if isinstance(data, dict):
            records = data.get("records", [])
        else:
            records = data

        if not isinstance(records, list):
            return []

        out: List[Dict[str, Any]] = []
        for item in records:
            if isinstance(item, dict):
                out.append(item)
        return out

    def list_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_records_unsafe()

    def append_record(self, record: Dict[str, Any]) -> None:
        with self._lock:
            records = self._read_records_unsafe()
            records.append(record)
            self.path.write_text(
                json.dumps({"records": records}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def clear(self) -> None:
        with self._lock:
            self.path.write_text(
                json.dumps({"records": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def build_prompt(records: List[Dict[str, Any]]) -> str:
        selected = records[-20:]

        lines: List[str] = [
            "Ты помощник, который преобразует распознанные жесты в понятный русский текст.",
            "Ниже сохранённые записи распознавания жестов. Каждая запись содержит предложение (если собрано) и последовательность жестов.",
            "Сформируй итоговый текст по последней записи, а затем предложи краткую интерпретацию/перефразирование.",
            "",
        ]

        for idx, rec in enumerate(selected, start=1):
            created_at = rec.get("created_at", "")
            lines.append(f"Запись {idx}: {created_at}")

            sentence = rec.get("sentence")
            if isinstance(sentence, str) and sentence.strip():
                lines.append(f"Предложение: {sentence.strip()}")

            events = rec.get("events", [])
            if isinstance(events, list) and events:
                gestures: List[str] = []
                for ev in events:
                    if isinstance(ev, dict):
                        g = ev.get("gesture")
                        if isinstance(g, str) and g:
                            gestures.append(g)
                if gestures:
                    lines.append("Жесты: " + ", ".join(gestures))

            lines.append("")

        lines.append("Ответ дай на русском.")
        return "\n".join(lines)
