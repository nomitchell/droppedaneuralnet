import json
import sys
from evaluate_core import evaluate_from_cli_arg, fail


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps(fail("Usage: python evaluate.py <comma-separated permutation>")))
        sys.exit(0)

    result_json = evaluate_from_cli_arg(sys.argv[1])
    print(result_json)
    try:
        result = json.loads(result_json)
        if result.get("solved", False):
            print("SOLVED", file=sys.stderr)
    except Exception:
        pass