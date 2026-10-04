"""Verify IBM Quantum credentials before spending queue time.

Checks the connection, lists reachable backends, and reports what the cheapest useful
Q-Peptide circuit would cost on each — so you can see whether a run is worth submitting
before you submit it.

Reads credentials from the environment only and never prints them.

    ./.venv/bin/python -m backend.check_ibm
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


from backend.utils.env import load_dotenv as _load_dotenv


def main() -> int:
    loaded = _load_dotenv()
    print("IBM Quantum connection check")
    print(f"  .env loaded: {loaded or 'nothing (file missing or already in environment)'}\n")

    from backend.optimization.hardware import (
        IBM_INSTANCE_VARS,
        IBM_TOKEN_VARS,
        get_ibm_service,
        ibm_credentials_available,
    )

    has_token = ibm_credentials_available()
    has_inst = any(os.environ.get(v) for v in IBM_INSTANCE_VARS)
    print(f"  API key   ({' / '.join(IBM_TOKEN_VARS)}): "
          f"{'present' if has_token else 'MISSING'}")
    print(f"  instance  ({' / '.join(IBM_INSTANCE_VARS)}): "
          f"{'present' if has_inst else 'not set'}")

    if not has_token:
        print("\n  -> Add your IBM Cloud API key to .env as:")
        print("     QISKIT_IBM_TOKEN=<your key>")
        print("     The key value is shown ONLY once, when you create it. If you did not")
        print("     save it, create a new one in IBM Cloud -> API keys.")
        return 1

    print("\n  connecting…")
    service, diag = get_ibm_service(explain=True)

    for attempt in diag.get("attempts", []):
        status = "ok" if attempt["ok"] else "failed"
        print(f"    channel {attempt['channel']:<22} {status}")
        if not attempt["ok"]:
            print(f"      {attempt['error'][:200]}")

    if service is None:
        print(f"\n  CONNECTION FAILED: {diag.get('error')}")
        if diag.get("hint"):
            print(f"\n  {diag['hint']}")
        return 1

    print(f"\n  CONNECTED via channel '{diag.get('channel_used')}'\n")

    try:
        backends = service.backends(operational=True, simulator=False)
    except Exception as exc:
        print(f"  could not list backends: {type(exc).__name__}: {exc}")
        return 1

    if not backends:
        print("  no operational hardware backends visible on this account.")
        return 1

    print(f"  {len(backends)} operational backend(s):\n")
    print(f"    {'backend':<22}{'qubits':>8}{'pending jobs':>14}")
    print("    " + "-" * 44)
    rows = []
    for b in backends:
        name = b.name if isinstance(b.name, str) else b.name()
        try:
            pending = b.status().pending_jobs
        except Exception:
            pending = None
        nq = getattr(b, "num_qubits", None)
        rows.append((name, nq, pending))
        print(f"    {name:<22}{str(nq):>8}{str(pending):>14}")

    usable = [r for r in rows if r[1] and r[1] >= 8]
    print(f"\n  backends with >= 8 qubits (enough for the small config): {len(usable)}")

    print("\n  Recommended first run — smallest config with real signal:")
    print("    N = 6 candidate mutations -> 8 qubits, p = 1 -> ~56 two-qubit gates")
    print("    (the headline N=14 p=3 config is ~720 two-qubit gates and will return noise)")
    print("\n  Run it with:")
    print("    ./.venv/bin/python -m backend.experiments.hardware_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
