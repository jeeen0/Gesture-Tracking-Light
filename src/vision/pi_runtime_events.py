import json
import time


def emit(kind, **data):
    data["type"] = kind
    data["ts"] = round(time.time(), 3)
    print("EVENT " + json.dumps(data, ensure_ascii=False), flush=True)


def emit_state(payload):
    print("STATE " + json.dumps(payload, ensure_ascii=False), flush=True)
