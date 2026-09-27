"""Probe public source endpoints without downloading datasets.

This checks reachability only. HTTP reachability never implies data quality,
licensing permission, historical PIT validity, or production eligibility.
"""
from __future__ import annotations

import argparse
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.research.opta_like_source_registry import all_sources

UA = "Soccer-Prediction-Research/source-probe/1.0"
TIMEOUT_SECONDS = 15


def _probe(url: str) -> dict:
    request = Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/json,*/*"}, method="HEAD")
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS, context=ssl.create_default_context()) as response:
            return {"status":"REACHABLE","http_status":int(response.status),
                    "content_type":str(response.headers.get("Content-Type","")),
                    "final_url":str(response.geturl()),"method":"HEAD"}
    except HTTPError as exc:
        if exc.code not in {405,403,501}:
            return {"status":"HTTP_ERROR","http_status":int(exc.code),"error":str(exc)}
    except (URLError, TimeoutError) as exc:
        return {"status":"NETWORK_ERROR","error":f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}

    get_request = Request(url, headers={"User-Agent":UA,"Accept":"text/html,application/json,*/*","Range":"bytes=0-1023"}, method="GET")
    try:
        with urlopen(get_request, timeout=TIMEOUT_SECONDS, context=ssl.create_default_context()) as response:
            return {"status":"REACHABLE","http_status":int(response.status),
                    "content_type":str(response.headers.get("Content-Type","")),
                    "final_url":str(response.geturl()),"method":"GET_RANGE"}
    except HTTPError as exc:
        return {"status":"HTTP_ERROR","http_status":int(exc.code),"error":str(exc)}
    except (URLError, TimeoutError) as exc:
        return {"status":"NETWORK_ERROR","error":f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}


def probe_all() -> dict:
    checked_at = datetime.now(timezone.utc).isoformat()
    results = []
    for source in all_sources():
        results.append({
            "key": source.key, "name": source.name, "url": source.source_url,
            "access": source.access, "pit_status": source.pit_status,
            "license_status": source.license_status,
            "prediction_value": source.prediction_value, "priority": source.priority,
            **_probe(source.source_url),
        })
    reachable = sum(x["status"] == "REACHABLE" for x in results)
    return {
        "schema_version": 1, "checked_at_utc": checked_at, "total": len(results),
        "reachable": reachable, "unreachable_or_error": len(results)-reachable,
        "results": results,
        "important_notice": "Reachability is not evidence of PIT safety, license permission, completeness, historical publication timing, or production eligibility.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/opta_like_source_probe.json")
    args = parser.parse_args()
    result = probe_all()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"OK","total":result["total"],"reachable":result["reachable"],
                      "unreachable_or_error":result["unreachable_or_error"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
