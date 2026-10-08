"""Readiness probe for the Phase E ROMBRL reproduction environment.

Imports each dependency defensively, records its version (or the import error), and
writes a machine-readable ``readiness.json``. Exit code is 0 only when every *critical*
component imports, so ``provision.sh`` can use it as the automated go/no-go gate.

Critical (needed for D4RL MuJoCo): torch, numpy, gym, mujoco, d4rl.
Optional (reported, not gating): gymnasium, dm_control, rliable, mujoco_py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

CRITICAL = ("torch", "numpy", "gym", "mujoco", "d4rl")
OPTIONAL = ("gymnasium", "dm_control", "rliable", "mujoco_py")


def _probe(module: str) -> dict:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure is reported verbatim
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": getattr(mod, "__version__", "unknown"),
        "error": None,
    }


def build_report(vendor: Path) -> dict:
    components = {name: _probe(name) for name in CRITICAL + OPTIONAL}
    critical_ok = all(components[name]["available"] for name in CRITICAL)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "critical": list(CRITICAL),
        "optional": list(OPTIONAL),
        "components": components,
        "rombri_vendor": {
            "path": str(vendor),
            "present": (vendor / "D4RL").is_dir() or (vendor / "Fusion").is_dir(),
        },
        "ready": critical_ok,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="readiness.json", help="path for the JSON report")
    ap.add_argument("--vendor", default="", help="path to the vendored ROMBRL checkout")
    args = ap.parse_args(argv)

    report = build_report(Path(args.vendor) if args.vendor else Path("."))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in CRITICAL + OPTIONAL:
        info = report["components"][name]
        tag = "ok" if info["available"] else "MISSING"
        detail = info["version"] if info["available"] else info["error"]
        gate = "critical" if name in CRITICAL else "optional"
        print(f"[verify] {name:10s} [{gate:8s}] {tag:7s} {detail}")

    print(f"[verify] rombri vendor present: {report['rombri_vendor']['present']}")
    print(f"[verify] READY={report['ready']} -> {out}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
