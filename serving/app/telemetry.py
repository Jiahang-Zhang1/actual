from __future__ import annotations

import os
import platform
import socket

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def hardware_string() -> str:
    cpu_count = os.cpu_count() or 0
    mem_gb = None
    if psutil is not None:
        try:
            mem_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            mem_gb = None

    parts = [
        f"host={socket.gethostname()}",
        f"platform={platform.platform()}",
        f"cpus={cpu_count}",
    ]
    if mem_gb is not None:
        parts.append(f"memory_gb={mem_gb}")
    return "; ".join(parts)
