"""
test_request.py - send three example orders to the running API and print the results.

Start the server first (in another terminal):
    uvicorn main:app --reload
Then run:
    python test_request.py
"""

import json
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------------------
# Three example orders
# ---------------------------------------------------------------------------------------
# Each is (label, expected bucket, JSON body). Fields after "is_returned" are the
# optional history lookups a real backend would fetch from its database.

LOW_RISK = {
    # A loyal 4-year-old account buying an ordinary clothing item in the evening,
    # shipping to their own billing address, on their own device.
    "customer_id": "CUST-900001",
    "order_amount": 1799,
    "product_category": "clothing",
    "timestamp": "2026-09-11T19:45:00",
    "account_creation_date": "2022-06-15",
    "shipping_address_id": "ADDR-900001",
    "billing_address_id": "ADDR-900001",
    "device_id": "DEV-900001",
    "ip_address": "117.196.44.12",
    "quantity": 1,
    "is_returned": 0,
    "customer_order_count": 14,
    "customer_return_rate": 0.07,
    "orders_last_24h": 0,
    "address_reuse_count": 1,
    "device_reuse_count": 1,
    "ip_reuse_count": 1,
    "address_device_pair_count": 1,
    "customer_avg_amount": 2100,
}

HIGH_RISK = {
    # A 1-day-old account buying a ~3x-average electronics order at 2:40 AM, billing to a
    # different address, from a device and address shared with 6-7 other accounts that
    # were all created within 4 days of each other. This is the "fraud ring" pattern.
    "customer_id": "CUST-900002",
    "order_amount": 94999,
    "product_category": "electronics",
    "timestamp": "2026-09-11T02:40:00",
    "account_creation_date": "2026-09-10",
    "shipping_address_id": "ADDR-900002",
    "billing_address_id": "ADDR-900099",
    "device_id": "DEV-900002",
    "ip_address": "157.32.76.84",
    "quantity": 1,
    "is_returned": 0,
    "customer_order_count": 1,
    "orders_last_24h": 0,
    "address_reuse_count": 6,
    "device_reuse_count": 7,
    "ip_reuse_count": 7,
    "address_device_pair_count": 6,
    "cluster_signup_window_days": 4,
}

MEDIUM_RISK = {
    # An established 14-month-old account placing a normal home & kitchen order at 2 PM,
    # but from a device used by 6 different accounts - a shared office or cyber-cafe
    # computer, say. Suspicious enough to look at, not enough to block.
    "customer_id": "CUST-900003",
    "order_amount": 2499,
    "product_category": "home_kitchen",
    "timestamp": "2026-09-11T14:10:00",
    "account_creation_date": "2025-07-20",
    "shipping_address_id": "ADDR-900003",
    "billing_address_id": "ADDR-900003",
    "device_id": "DEV-900003",
    "ip_address": "103.87.12.201",
    "quantity": 1,
    "is_returned": 0,
    "customer_order_count": 6,
    "customer_return_rate": 0.0,
    "orders_last_24h": 0,
    "address_reuse_count": 1,
    "device_reuse_count": 6,
    "ip_reuse_count": 1,
    "address_device_pair_count": 1,
    "customer_avg_amount": 2300,
}

EXAMPLES = [
    ("LOW RISK   - loyal customer, normal order", "Low", LOW_RISK),
    ("HIGH RISK  - new account, big order, shared device + address", "High", HIGH_RISK),
    ("MEDIUM RISK - established account on a shared device", "Medium", MEDIUM_RISK),
]


def check_server() -> None:
    """Fail early with a helpful message if the server isn't running or the model
    didn't load."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        sys.exit(f"Could not connect to {BASE_URL}.\n"
                 "Start the server first:  uvicorn main:app --reload")

    body = r.json()
    if r.status_code != 200:
        sys.exit(f"Server is up but the model is NOT loaded:\n{json.dumps(body, indent=2)}")

    print(f"Server OK - model v{body['model_version']} trained {body['trained_at']}")
    for warning in body.get("warnings", []):
        print(f"  warning: {warning}")


def main() -> None:
    check_server()
    results = []

    for label, expected, payload in EXAMPLES:
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)

        # requests.post(..., json=payload) converts the dict to JSON and sets the
        # Content-Type header for us.
        response = requests.post(f"{BASE_URL}/score-order", json=payload, timeout=10)
        print(f"HTTP {response.status_code}")

        if response.status_code != 200:
            # 422 = validation error; the body explains exactly which field was wrong.
            print(json.dumps(response.json(), indent=2))
            results.append((label, expected, "ERROR"))
            continue

        data = response.json()
        print(f"risk_score      : {data['risk_score']}")
        print(f"risk_bucket     : {data['risk_bucket']}   (expected {expected})")
        print(f"anomaly_score   : {data['anomaly_score']}")
        print(f"flags_triggered : {data['flags_triggered'] or '(none)'}")
        print(f"explanation     : {data['explanation']}")
        results.append((label, expected, data["risk_bucket"]))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for label, expected, got in results:
        mark = "PASS" if got == expected else "CHECK"
        print(f"[{mark}] expected {expected:6s} got {got:6s}  {label}")


if __name__ == "__main__":
    main()
