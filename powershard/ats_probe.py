"""ATS attribute probe; UVA and copy speed alone do not establish ATS."""
import json
import sys
from .ats_memory import ats_capabilities


def probe_ats(index=0):
    return ats_capabilities(index)


if __name__ == "__main__":
    try:
        result = probe_ats(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    except Exception as error:
        result = dict(status="FAIL", reason=f"{type(error).__name__}: {error}")
    print(json.dumps(result))
