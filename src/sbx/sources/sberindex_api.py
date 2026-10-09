"""Клиент открытого API портала sberindex.ru.

Дашборды портала запрашивают данные через POST /api/sowa: исходный GET-маршрут
(например, `/dataset/v1/consumer-spending`) передаётся в base64, ответ приходит в
типизированной обёртке (`__string__<base64>`, `__number__<x>`, объекты key/value).
Сертификат сайта выдан НУЦ Минцифры, поэтому по умолчанию verify=False.
"""
from __future__ import annotations

import base64
import re
import time
import uuid

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
API_URL = "https://sberindex.ru/api/sowa"
_TYPED = re.compile(r"^__([a-z0-9]*)__(.*)$", re.S)


def _decode(node, hint=None):
    if isinstance(node, list):
        return [_decode(x) for x in node]
    if isinstance(node, dict) and node.get("type") == "object":
        out = {}
        for item in node["value"]:
            if item.get("type") == "longstring":
                return "".join(_decode(item["value"]))
            out[item["key"]] = _decode(item["value"], item.get("type"))
        return out
    if isinstance(node, dict):
        return {k: _decode(v) for k, v in node.items()}
    m = _TYPED.match(str(node))
    if not m:
        return node
    kind = hint if hint and hint != "object" else m.group(1)
    val = m.group(2)
    if kind == "null":
        return None
    if kind == "number":
        return float(val)
    if kind == "boolean":
        return val == "true"
    if kind == "string":
        return base64.b64decode(val).decode("utf-8")
    return val


class SberIndexAPI:
    def __init__(self, verify=False, timeout=90, pause=0.3):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (research; sberindex-task2)"
        self.verify, self.timeout, self.pause = verify, timeout, pause

    def get(self, route: str):
        body = {"SOWA": {"method": "GET", "route": base64.b64encode(route.encode()).decode(),
                         "data": {"type": "object", "value": []}}}
        for attempt in range(4):
            try:
                r = self.s.post(API_URL, json=body, headers={"rquid": uuid.uuid4().hex},
                                verify=self.verify, timeout=self.timeout)
                r.raise_for_status()
                p = r.json()
                time.sleep(self.pause)
                return _decode(p["SOWA"]["data"]) if "SOWA" in p else p
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def dataset(self, name: str, query: str = "") -> pd.DataFrame:
        """Набор целиком как таблица (все регионы/разрезы, все периоды)."""
        data = self.get(f"/dataset/v1/{name}{query}")
        rows = data.get("data", data) if isinstance(data, dict) else data
        return pd.DataFrame(rows)
