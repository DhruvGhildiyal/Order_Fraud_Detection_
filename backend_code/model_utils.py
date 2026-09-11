"""
model_utils.py - feature engineering and risk scoring for the fraud API.

How the two files split the work:
    main.py         handles HTTP: receiving requests, validating them, sending responses.
    model_utils.py  handles the maths: turning an order into features, running the model,
                    applying the risk rules.

Keeping them apart means you can test the scoring logic without starting a web server
(just `import model_utils`), and the web layer never needs to know how a feature is built.

IMPORTANT - every calculation below mirrors the training notebook. If you change a feature
or a rule in the notebook, change it here too. Otherwise the API quietly scores orders
differently from how the model was evaluated. That mismatch has a name, "training/serving
skew", and it is one of the most common ways ML systems break in production.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


# ---------------------------------------------------------------------------------------
# Rule thresholds
# ---------------------------------------------------------------------------------------
# The bundle stores the risk *weights* and the Low/Medium/High cut-offs, but the numbers
# inside each rule ("account <= 3 days old", "device shared by 5+ accounts") were written
# directly in the notebook, so they are copied here. Keep them in sync with the notebook.
RULES = {
    "new_account_max_age_days": 3,         # new_account_high_value_flag: account this new...
    "high_value_min_ratio": 2.0,           # ...AND order >= 2x its category average
    "odd_hour_start": 1,                   # is_odd_hour covers 01:00 ...
    "odd_hour_end": 5,                     # ... up to, but not including, 05:00
    "odd_hour_max_account_age_days": 14,   # odd_hour_new_account_flag
    "shared_entity_min_accounts": 5,       # shared_device / shared_address / shared_ip flags
    "velocity_min_orders_24h": 3,          # velocity_flag
    "return_abuse_min_rate": 0.5,          # return_abuse_flag: returns >= 50% of orders...
    "return_abuse_min_orders": 5,          # ...across 5+ orders...
    "return_abuse_min_avg_amount": 8000.0, # ...with an expensive average basket
}

# Every flag this module knows how to compute. Used to check the bundle's risk_weights.
KNOWN_FLAGS = [
    "coordinated_cluster_flag", "new_account_high_value_flag", "return_abuse_flag",
    "shared_device_flag", "odd_hour_new_account_flag", "shared_address_flag",
    "isolation_forest_flag", "shared_ip_flag", "velocity_flag", "billing_mismatch",
    "iqr_flag",
]


# ---------------------------------------------------------------------------------------
# Loading the model bundle
# ---------------------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Everything loaded from the joblib file, with the thresholds normalised.

    A dataclass is just a class that holds data. Using one instead of passing the raw
    dict around means a typo like `bundle.low_mde` fails loudly instead of returning None.
    """

    model: Any                                  # the fitted IsolationForest
    feature_names: list[str]                    # column order the model expects
    category_stats: dict[str, dict[str, float]] # {"electronics": {"mean_amount":..., "std_amount":...}}
    risk_weights: dict[str, float]              # {"shared_device_flag": 20, ...}
    low_med: float                              # score >= this -> Medium
    med_high: float                             # score >= this -> High
    score_cap: float = 100.0                    # risk_score never exceeds this
    iqr_deviation_upper: float | None = None    # iqr_flag threshold, in std deviations
    cluster_min_accounts: int = 3               # coordinated_cluster_flag: accounts on one address+device
    cluster_tight_signup_days: float = 14.0     # ...all created within this many days
    metadata: dict[str, Any] = field(default_factory=dict)
    load_warnings: list[str] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return sorted(self.category_stats)


def load_bundle(path: str | Path) -> ModelBundle:
    """Load the joblib bundle from disk and check it has everything we need.

    Raises a clear error if the file is missing or incomplete, so a bad path shows up
    at startup instead of as a confusing crash on the first request.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at: {path}")

    # joblib.load un-pickles the file. If the model was saved with a different
    # scikit-learn version, sklearn emits a noisy warning for every tree in the forest.
    # We silence those here and report the mismatch once, clearly, below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = joblib.load(path)

    missing = [k for k in ("model", "feature_names", "category_stats", "risk_weights")
               if k not in raw]
    if missing:
        raise KeyError(f"Model bundle is missing required keys: {missing}")

    # The cut-offs may be stored either as {"bucket_thresholds": {"low_medium": ..,
    # "medium_high": ..}} or as top-level LOW_MED / MED_HIGH keys. Accept both.
    if "bucket_thresholds" in raw:
        low_med = raw["bucket_thresholds"]["low_medium"]
        med_high = raw["bucket_thresholds"]["medium_high"]
    elif "LOW_MED" in raw and "MED_HIGH" in raw:
        low_med, med_high = raw["LOW_MED"], raw["MED_HIGH"]
    else:
        raise KeyError("Model bundle has no bucket thresholds "
                       "(expected 'bucket_thresholds' or 'LOW_MED'/'MED_HIGH').")

    model = raw["model"]
    feature_names = list(raw["feature_names"])
    n_expected = getattr(model, "n_features_in_", len(feature_names))
    if n_expected != len(feature_names):
        raise ValueError(f"Model expects {n_expected} features but bundle lists "
                         f"{len(feature_names)} feature names.")

    load_warnings: list[str] = []
    trained_with = raw.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        load_warnings.append(
            f"Model was trained with scikit-learn {trained_with} but "
            f"{sklearn.__version__} is installed. Run: pip install "
            f"scikit-learn=={trained_with}")

    unknown = sorted(set(raw["risk_weights"]) - set(KNOWN_FLAGS))
    if unknown:
        load_warnings.append(f"risk_weights contains flags this API does not compute "
                             f"(they will never fire): {unknown}")

    cluster_rules = raw.get("cluster_rules", {})
    return ModelBundle(
        model=model,
        feature_names=feature_names,
        category_stats=raw["category_stats"],
        risk_weights=dict(raw["risk_weights"]),
        low_med=float(low_med),
        med_high=float(med_high),
        score_cap=float(raw.get("risk_score_cap", 100)),
        iqr_deviation_upper=raw.get("iqr_bounds", {}).get("deviation_upper"),
        cluster_min_accounts=int(cluster_rules.get("min_accounts", 3)),
        cluster_tight_signup_days=float(cluster_rules.get("tight_signup_days", 14)),
        metadata={k: raw.get(k) for k in
                  ("model_type", "version", "trained_at", "trained_on_rows",
                   "sklearn_version", "contamination")},
        load_warnings=load_warnings,
    )


# ---------------------------------------------------------------------------------------
# Step 1 - feature engineering
# ---------------------------------------------------------------------------------------
def build_features(order: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """Turn one raw order into the engineered features, exactly as the notebook did.

    `order` is a plain dict (main.py converts the validated request into one). It must
    contain the raw order fields, and MAY contain the history lookups - anything missing
    falls back to "brand-new customer / device / address".

    Returns a dict holding the 15 model features plus a few helper values the rules and
    the explanation need (those helpers are never fed to the model).
    """
    ts: datetime = order["timestamp"]
    if ts.tzinfo is not None:
        # Training timestamps had no timezone. Drop it and keep the wall-clock time,
        # so "02:30+05:30" is treated as 02:30 - the hour the customer actually saw.
        ts = ts.replace(tzinfo=None)

    created = order["account_creation_date"]
    if isinstance(created, datetime):
        created = created.date()

    amount = float(order["order_amount"])
    stats = bundle.category_stats[order["product_category"]]
    cat_mean = float(stats["mean_amount"])
    cat_std = float(stats["std_amount"]) or 1.0     # guard against dividing by zero

    hour = ts.hour
    account_age_days = (ts.date() - created).days   # notebook: order_date - creation_date

    features: dict[str, Any] = {
        # ---- computed from the order itself ------------------------------------------
        "account_age_days": account_age_days,
        "log_order_amount": math.log1p(amount),            # log(1 + amount), tames skew
        "order_amount_deviation": (amount - cat_mean) / cat_std,  # z-score vs category
        "quantity": int(order.get("quantity", 1)),
        "is_odd_hour": int(RULES["odd_hour_start"] <= hour < RULES["odd_hour_end"]),
        "hour_of_day": hour,
        "day_of_week": ts.weekday(),                       # Monday = 0, same as pandas
        "billing_mismatch": int(order["billing_address_id"] != order["shipping_address_id"]),

        # ---- history lookups: supplied by the caller, default = never seen before ----
        "orders_last_24h": int(order.get("orders_last_24h") or 0),
        "customer_return_rate": float(order.get("customer_return_rate") or 0.0),
        "customer_order_count": int(order.get("customer_order_count") or 0),
        "address_reuse_count": int(order.get("address_reuse_count") or 0),
        "device_reuse_count": int(order.get("device_reuse_count") or 0),
        "ip_reuse_count": int(order.get("ip_reuse_count") or 0),
        "address_device_pair_count": int(order.get("address_device_pair_count") or 0),
    }

    # ---- helpers for the rules and the explanation (NOT model inputs) -----------------
    features["amount_vs_category_avg"] = amount / cat_mean if cat_mean else 0.0
    features["product_category"] = order["product_category"]
    features["order_amount"] = amount
    # A brand-new customer's average basket is simply this order.
    avg = order.get("customer_avg_amount")
    features["customer_avg_amount"] = float(avg) if avg is not None else amount
    # Unknown unless the database tells us; None means "can't judge coordination".
    features["cluster_signup_window_days"] = order.get("cluster_signup_window_days")
    return features


# ---------------------------------------------------------------------------------------
# Step 2 - the Isolation Forest
# ---------------------------------------------------------------------------------------
def run_isolation_forest(features: dict[str, Any], bundle: ModelBundle) -> tuple[float, int]:
    """Return (anomaly_score, isolation_forest_flag) for one order.

    The feature vector is built in bundle.feature_names order - never a hand-typed list -
    because a model only sees column positions, not names. Swap two columns and it will
    happily return a confident, meaningless score.
    """
    X = np.array([[float(features[name]) for name in bundle.feature_names]])

    # score_samples: LOWER = more anomalous. We negate it so HIGHER = more suspicious,
    # which is the same convention the notebook used for its anomaly_score column.
    anomaly_score = float(-bundle.model.score_samples(X)[0])
    # predict returns -1 for "anomaly", 1 for "normal"
    is_anomaly = int(bundle.model.predict(X)[0] == -1)
    return anomaly_score, is_anomaly


# ---------------------------------------------------------------------------------------
# Step 3 - rules, score, bucket
# ---------------------------------------------------------------------------------------
def evaluate_rules(features: dict[str, Any], iso_flag: int, bundle: ModelBundle) -> dict[str, int]:
    """Check every risk rule. Returns {flag_name: 1 if it fired else 0}."""
    f = features
    window = f["cluster_signup_window_days"]
    shared = RULES["shared_entity_min_accounts"]

    return {
        # many accounts on one address + device, all created within a short window
        "coordinated_cluster_flag": int(
            f["address_device_pair_count"] >= bundle.cluster_min_accounts
            and window is not None
            and window <= bundle.cluster_tight_signup_days),
        # brand-new account buying something big for its category
        "new_account_high_value_flag": int(
            f["account_age_days"] <= RULES["new_account_max_age_days"]
            and f["amount_vs_category_avg"] >= RULES["high_value_min_ratio"]),
        # repeatedly orders and returns expensive items
        "return_abuse_flag": int(
            f["customer_return_rate"] >= RULES["return_abuse_min_rate"]
            and f["customer_order_count"] >= RULES["return_abuse_min_orders"]
            and f["customer_avg_amount"] >= RULES["return_abuse_min_avg_amount"]),
        "shared_device_flag": int(f["device_reuse_count"] >= shared),
        "odd_hour_new_account_flag": int(
            f["is_odd_hour"] == 1
            and f["account_age_days"] <= RULES["odd_hour_max_account_age_days"]),
        "shared_address_flag": int(f["address_reuse_count"] >= shared),
        "isolation_forest_flag": int(iso_flag),
        "shared_ip_flag": int(f["ip_reuse_count"] >= shared),
        "velocity_flag": int(f["orders_last_24h"] >= RULES["velocity_min_orders_24h"]),
        "billing_mismatch": int(f["billing_mismatch"]),
        "iqr_flag": int(bundle.iqr_deviation_upper is not None
                        and f["order_amount_deviation"] > bundle.iqr_deviation_upper),
    }


def compute_risk(flags: dict[str, int], bundle: ModelBundle) -> tuple[float, str]:
    """Add up the weights of every rule that fired, cap at 100, then bucket it."""
    raw = sum(weight * flags.get(name, 0) for name, weight in bundle.risk_weights.items())
    score = float(min(raw, bundle.score_cap))

    if score >= bundle.med_high:
        bucket = "High"
    elif score >= bundle.low_med:
        bucket = "Medium"
    else:
        bucket = "Low"
    return score, bucket


def _reason(flag: str, f: dict[str, Any]) -> str:
    """One short, human-readable sentence per flag."""
    reasons = {
        "coordinated_cluster_flag": lambda: (
            f"{f['address_device_pair_count']} accounts share this address and device, "
            f"created within {f['cluster_signup_window_days']:g} days"),
        "new_account_high_value_flag": lambda: (
            f"account is {f['account_age_days']} day(s) old and the order is "
            f"{f['amount_vs_category_avg']:.1f}x the {f['product_category']} average"),
        "return_abuse_flag": lambda: (
            f"customer returned {f['customer_return_rate']:.0%} of "
            f"{f['customer_order_count']} orders"),
        "shared_device_flag": lambda: f"device used by {f['device_reuse_count']} accounts",
        "odd_hour_new_account_flag": lambda: (
            f"placed at {f['hour_of_day']:02d}:00-{f['hour_of_day']:02d}:59 from a "
            f"{f['account_age_days']}-day-old account"),
        "shared_address_flag": lambda: (
            f"shipping address used by {f['address_reuse_count']} accounts"),
        "isolation_forest_flag": lambda: "unusual combination of order features (anomaly model)",
        "shared_ip_flag": lambda: f"IP address used by {f['ip_reuse_count']} accounts",
        "velocity_flag": lambda: f"{f['orders_last_24h']} earlier orders in the last 24h",
        "billing_mismatch": lambda: "billing address differs from shipping address",
        "iqr_flag": lambda: (
            f"order amount is {f['order_amount_deviation']:.1f} std devs above the "
            f"{f['product_category']} average"),
    }
    return reasons[flag]() if flag in reasons else flag


def build_explanation(score: float, bucket: str, fired: list[str], f: dict[str, Any]) -> str:
    """Summarise the top reasons (highest-weight flags first) in one line."""
    head = f"{bucket} risk (score {score:g}/100)"
    if not fired:
        return f"{head}: no risk rules fired."
    top = [_reason(flag, f) for flag in fired[:3]]
    more = f" (+{len(fired) - 3} more)" if len(fired) > 3 else ""
    return f"{head}: " + "; ".join(top) + more + "."


# ---------------------------------------------------------------------------------------
# The one function main.py calls
# ---------------------------------------------------------------------------------------
def score_order(order: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """Full pipeline: features -> Isolation Forest -> rules -> score -> explanation."""
    features = build_features(order, bundle)
    anomaly_score, iso_flag = run_isolation_forest(features, bundle)
    flags = evaluate_rules(features, iso_flag, bundle)
    risk_score, risk_bucket = compute_risk(flags, bundle)

    # List the rules that fired, most important (highest weight) first.
    fired = sorted((name for name, on in flags.items()
                    if on and bundle.risk_weights.get(name, 0) > 0),
                   key=lambda name: -bundle.risk_weights[name])

    return {
        "risk_score": risk_score,
        "risk_bucket": risk_bucket,
        "anomaly_score": round(anomaly_score, 4),
        "flags_triggered": fired,
        "explanation": build_explanation(risk_score, risk_bucket, fired, features),
    }
