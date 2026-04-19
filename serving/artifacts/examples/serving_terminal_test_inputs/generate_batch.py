import json
import sys
from pathlib import Path

SAMPLES = [
    {"transaction_description": "STARBUCKS STORE 1458 NEW YORK NY", "country": "US", "currency": "USD"},
    {"transaction_description": "PAYROLL DIRECT DEPOSIT ACME INC", "country": "US", "currency": "USD"},
    {"transaction_description": "COMCAST CABLE PAYMENT", "country": "US", "currency": "USD"},
    {"transaction_description": "DONATION TO RED CROSS", "country": "US", "currency": "USD"},
    {"transaction_description": "AMAZON MARKETPLACE ORDER 114-1234567", "country": "US", "currency": "USD"},
    {"transaction_description": "MONTHLY ACCOUNT SERVICE FEE", "country": "US", "currency": "USD"},
    {"transaction_description": "UBER TRIP HELP.UBER.COM", "country": "US", "currency": "USD"},
    {"transaction_description": "CITY ELECTRIC BILL PAYMENT", "country": "US", "currency": "USD"},
]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("batch_generated.json")
items = [SAMPLES[i % len(SAMPLES)] for i in range(n)]
out.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")
print(f"wrote {out} with {n} items")
