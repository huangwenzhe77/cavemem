"""Isolated grader. Its stdin contains only a fresh request ID and terminal artifact."""
import json
import sys


def grade(payload):
    family, values, result = (payload[key] for key in ("family", "input", "submitted"))
    if not isinstance(values, list) or not all(type(n) is int for n in values):
        raise ValueError("malformed task input")
    answers = {
        "ascending": sorted(values),
        "descending": sorted(values, reverse=True),
        "unique": sorted(set(values)),
        "positive": sorted(n for n in values if n > 0),
    }
    if result is not None and (not isinstance(result, list) or not all(type(n) is int for n in result)):
        raise ValueError("malformed submitted artifact")
    return int(result == answers[family])


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(json.dumps({"request_id": request["request_id"], "reward": grade(request)}))
