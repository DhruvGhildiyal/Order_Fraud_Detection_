"""
main.py - FastAPI app that scores e-commerce orders for fraud risk.

Run it (from the folder containing this file):
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/        the HTML dashboard (form + result card)
    http://127.0.0.1:8000/docs    interactive API documentation
    http://127.0.0.1:8000/api     API info as JSON

What FastAPI gives you for free, in case it's your first time:
  * Request validation - declare the shape of the JSON with a pydantic model and FastAPI
    rejects anything that doesn't match with a clear 422 error, before your code runs.
  * Automatic docs - /docs (Swagger UI) is generated from the code below. Every Field
    description and example you write shows up there.
  * JSON conversion - return a dict or pydantic model and it becomes a JSON response.
"""

import ipaddress
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from model_utils import load_bundle, score_order

# uvicorn's logger, so our messages appear in the same terminal as the server output
log = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------------------
# Where the model lives
# ---------------------------------------------------------------------------------------
# ADJUST THIS PATH IF NEEDED. It assumes this layout:
#     project/
#         backend/main.py          <- this file
#         models/fraud_isolation_forest.joblib
#
# The relative path is resolved against the folder THIS FILE is in, not the folder your
# terminal happens to be in - so the server finds the model no matter where you start it.
# You can also override it without editing code:  set MODEL_PATH=C:\path\to\bundle.joblib
MODEL_RELATIVE_PATH = "../models/fraud_isolation_forest.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH") or
                  Path(__file__).resolve().parent / MODEL_RELATIVE_PATH).resolve()


# ---------------------------------------------------------------------------------------
# Startup: load the model ONCE
# ---------------------------------------------------------------------------------------
# Loading an 800 KB model takes a moment. Doing it inside the endpoint would repeat that
# work on every single request. Instead we load it once when the server starts and keep
# it in `app.state`, a place FastAPI provides for sharing objects between requests.
#
# `lifespan` is FastAPI's startup/shutdown hook: code before `yield` runs at startup,
# code after `yield` runs at shutdown. (Older tutorials use @app.on_event("startup");
# that still works but is deprecated in favour of this.)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = None
    app.state.load_error = None
    try:
        app.state.bundle = load_bundle(MODEL_PATH)
        log.info("Model loaded from %s (%d features)",
                 MODEL_PATH, len(app.state.bundle.feature_names))
        for warning in app.state.bundle.load_warnings:
            log.warning(warning)
    except Exception as exc:
        # Don't crash the server - start anyway so /health can tell you what went wrong.
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        log.error("MODEL NOT LOADED - %s", app.state.load_error)

    yield   # the server runs while we're paused here

    log.info("Shutting down.")   # nothing to clean up, but this is where it would go


app = FastAPI(
    title="E-Commerce Fraud Scoring API",
    description="Scores a single order with an Isolation Forest plus weighted risk rules.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------------------
# Browsers block a web page on one origin (e.g. a React app on localhost:3000) from
# calling an API on another origin (localhost:8000) unless the API says it's allowed.
# "*" allows everyone - fine for local testing, NOT for production, where you would list
# your real frontend URL instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,     # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------------------
# Request and response shapes (pydantic models)
# ---------------------------------------------------------------------------------------
class OrderRequest(BaseModel):
    """The JSON body POST /score-order accepts.

    Each attribute is a field. The type annotation (str, float, datetime...) tells
    pydantic what to accept, and `Field(...)` adds constraints and documentation.
    `Field(...)` with `...` as the first argument means REQUIRED; any other first
    argument is the default value, which makes the field optional.
    """

    # ---- Raw order fields - known the moment the order is placed -------------------
    customer_id: str = Field(..., min_length=1, description="Customer identifier")
    order_amount: float = Field(..., gt=0, description="Order value in INR")
    product_category: str = Field(..., description="e.g. electronics, clothing, groceries")
    timestamp: datetime = Field(..., description="When the order was placed, e.g. 2026-09-11T14:30:00")
    account_creation_date: date = Field(..., description="YYYY-MM-DD")
    shipping_address_id: str = Field(..., min_length=1)
    billing_address_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    ip_address: str = Field(..., description="IPv4 or IPv6 address")
    quantity: int = Field(1, ge=1, le=1000)
    # Accepted for completeness, but NOT used for scoring: at the moment an order is
    # placed it cannot have been returned yet, and the model was not trained on it.
    is_returned: int = Field(0, ge=0, le=1)

    # ---- History lookups - OPTIONAL ------------------------------------------------
    # These describe the customer's past and how many OTHER accounts share this device,
    # address or IP. A single order can't tell you any of that. In production, the
    # backend would query the orders database before scoring, e.g.
    #     SELECT COUNT(DISTINCT customer_id) FROM orders WHERE device_id = :device_id
    # For testing you pass them in the request. If you leave them out they default to 0,
    # meaning "never seen this customer / device / address before".
    orders_last_24h: int = Field(0, ge=0, description="Customer's previous orders in the last 24h")
    customer_return_rate: float = Field(0.0, ge=0, le=1, description="Fraction of past orders returned")
    customer_order_count: int = Field(0, ge=0, description="Customer's total order count")
    address_reuse_count: int = Field(0, ge=0, description="Distinct accounts using this shipping address")
    device_reuse_count: int = Field(0, ge=0, description="Distinct accounts using this device")
    ip_reuse_count: int = Field(0, ge=0, description="Distinct accounts using this IP")
    address_device_pair_count: int = Field(0, ge=0, description="Distinct accounts using this address AND device")
    # Two more lookups two of the risk rules need. Without them those rules can't fire.
    cluster_signup_window_days: Optional[float] = Field(
        None, ge=0, description="Days between first and last signup among accounts sharing this address+device")
    customer_avg_amount: Optional[float] = Field(
        None, ge=0, description="Customer's average order value (defaults to this order's amount)")

    # ---- Custom validation ---------------------------------------------------------
    # @field_validator runs extra checks on one field after the type check passes.
    @field_validator("ip_address")
    @classmethod
    def ip_must_be_valid(cls, value: str) -> str:
        ipaddress.ip_address(value)     # raises ValueError -> FastAPI returns 422
        return value

    @field_validator("product_category")
    @classmethod
    def normalise_category(cls, value: str) -> str:
        return value.strip().lower()    # "Electronics " -> "electronics"

    # @model_validator(mode="after") runs once all fields are valid, so it can compare them.
    @model_validator(mode="after")
    def account_must_exist_before_order(self):
        if self.account_creation_date > self.timestamp.date():
            raise ValueError("account_creation_date cannot be after the order timestamp")
        return self

    # This example pre-fills the "Try it out" box in Swagger UI.
    model_config = {
        "json_schema_extra": {
            "examples": [{
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
                "device_reuse_count": 7,
                "address_reuse_count": 6,
                "ip_reuse_count": 7,
                "address_device_pair_count": 6,
                "cluster_signup_window_days": 4,
            }]
        }
    }


class ScoreResponse(BaseModel):
    """What POST /score-order returns. Declaring it documents the output in /docs
    and guarantees the endpoint can never return a differently-shaped response."""

    risk_score: float = Field(..., description="0-100, higher is riskier")
    risk_bucket: Literal["Low", "Medium", "High"]
    anomaly_score: float = Field(..., description="Isolation Forest score, higher = more anomalous")
    flags_triggered: list[str] = Field(..., description="Rules that fired, most important first")
    explanation: str


# ---------------------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------------------
# A decorator like @app.get("/") registers the function below it as the handler for
# that HTTP method + URL.

# response_class=HTMLResponse tells FastAPI "this is a web page, not JSON": it sets the
# Content-Type header to text/html, so the browser renders the page instead of showing
# raw text. include_in_schema=False keeps it out of /docs, which documents the JSON API.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    """Serve the HTML dashboard. DASHBOARD_HTML is defined at the bottom of this file."""
    return DASHBOARD_HTML


@app.get("/api")
def api_info():
    """Basic information about the API, as JSON. (This used to live at "/".)"""
    return {
        "name": app.title,
        "version": app.version,
        "description": app.description,
        "endpoints": {
            "GET /": "HTML dashboard",
            "GET /api": "this page",
            "GET /health": "is the model loaded?",
            "POST /score-order": "score one order",
            "GET /docs": "interactive Swagger UI",
        },
    }


@app.get("/health")
def health():
    """Confirms the model is loaded. Returns HTTP 503 if it isn't."""
    bundle = app.state.bundle
    if bundle is None:
        # 503 = "Service Unavailable": the server is up but can't do its job yet.
        return JSONResponse(status_code=503, content={
            "status": "error",
            "model_loaded": False,
            "model_path": str(MODEL_PATH),
            "error": app.state.load_error,
        })
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "model_version": bundle.metadata.get("version"),
        "trained_at": bundle.metadata.get("trained_at"),
        "n_features": len(bundle.feature_names),
        "categories": bundle.categories,
        "thresholds": {"low_medium": bundle.low_med, "medium_high": bundle.med_high},
        "sklearn_trained": bundle.metadata.get("sklearn_version"),
        "sklearn_running": sklearn.__version__,
        "warnings": bundle.load_warnings,
    }


# Note: this is a plain `def`, not `async def`. Running the model is CPU work, and
# FastAPI runs plain `def` endpoints in a thread pool so one slow request can't freeze
# the whole server. Use `async def` when your code awaits I/O (databases, HTTP calls).
@app.post("/score-order", response_model=ScoreResponse)
def score_order_endpoint(order: OrderRequest):
    """Score a single order for fraud risk.

    By the time this function runs, FastAPI has already parsed the JSON and validated
    it against OrderRequest. If anything was wrong, the client got a 422 and we never
    got here.
    """
    bundle = app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503,
                            detail=f"Model not loaded: {app.state.load_error}")

    # The model only has statistics for categories it saw in training. Scoring an
    # unknown one would mean inventing numbers, so we refuse with a helpful message.
    if order.product_category not in bundle.category_stats:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown product_category '{order.product_category}'. "
                   f"Valid categories: {bundle.categories}")

    try:
        # model_dump() turns the pydantic object into a plain dict for model_utils.
        return score_order(order.model_dump(), bundle)
    except Exception as exc:
        log.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")


# ---------------------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------------------
# The whole dashboard is one string: HTML, CSS and JavaScript together. FastAPI sends it
# as-is through HTMLResponse. No template engine (and therefore no jinja2) is needed,
# because Python doesn't fill anything into the page - all the dynamic work happens in
# the browser: JavaScript reads the form, calls POST /score-order with fetch(), and draws
# the result card. Because the page and the API come from the same server (same
# "origin"), the browser needs no CORS permission for that call.
#
# It's a raw string (r"""...""") so backslashes inside the JavaScript are left untouched.
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Risk Scoring</title>
<style>
  :root {
    --bg: #f4f6fb; --card: #ffffff; --ink: #0f172a; --muted: #64748b; --line: #e2e8f0;
    --brand: #4f46e5; --brand-dark: #4338ca;
    --low: #15803d;  --low-bg: #f0fdf4;  --low-line: #bbf7d0;
    --med: #c2410c;  --med-bg: #fff7ed;  --med-line: #fed7aa;
    --high: #dc2626; --high-bg: #fef2f2; --high-line: #fecaca;
    --radius: 16px;
    --shadow: 0 1px 2px rgba(15, 23, 42, .06), 0 10px 30px rgba(15, 23, 42, .08);
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }

  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
           gap: 12px; margin-bottom: 24px; }
  h1 { margin: 0; font-size: 28px; letter-spacing: -.02em; }
  .sub { margin: 4px 0 0; color: var(--muted); }
  .links { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .links a { color: var(--brand); text-decoration: none; font-weight: 600; font-size: 14px;
             padding: 6px 10px; border-radius: 8px; }
  .links a:hover { background: #eef2ff; }
  .pill { font-size: 13px; font-weight: 600; padding: 5px 12px; border-radius: 999px;
          background: var(--line); color: var(--muted); }
  .pill.ok  { background: var(--low-bg);  color: var(--low); }
  .pill.bad { background: var(--high-bg); color: var(--high); }

  .card { background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px; }

  .presets { display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
             padding-bottom: 20px; margin-bottom: 20px; border-bottom: 1px solid var(--line); }
  .presets span { color: var(--muted); font-size: 14px; margin-right: 2px; }
  .preset { display: inline-flex; align-items: center; gap: 8px; cursor: pointer;
            font: inherit; font-weight: 600; color: var(--ink); background: #fff;
            border: 1px solid var(--line); border-radius: 10px; padding: 8px 14px;
            transition: border-color .15s, box-shadow .15s; }
  .preset::before { content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--dot); }
  .preset:hover { border-color: var(--dot); box-shadow: 0 2px 8px rgba(15, 23, 42, .08); }

  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 18px; }
  label { display: block; font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px; }
  input, select { width: 100%; font: inherit; color: var(--ink); background: #fff;
                  padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; }
  input:focus, select:focus { outline: none; border-color: var(--brand);
                              box-shadow: 0 0 0 3px rgba(79, 70, 229, .15); }
  .check { display: flex; align-items: center; gap: 10px; padding-top: 26px; }
  .check input { width: 18px; height: 18px; accent-color: var(--brand); }
  .check label { margin: 0; font-size: 14px; }

  details { margin-top: 22px; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 2px 16px; }
  details[open] { padding-bottom: 18px; }
  summary { cursor: pointer; font-weight: 600; padding: 10px 0; }
  .hint { color: var(--muted); font-size: 13px; margin: 0 0 14px; }

  .actions { margin-top: 22px; }
  .primary { cursor: pointer; font: inherit; font-weight: 700; color: #fff; background: var(--brand);
             border: 0; border-radius: 10px; padding: 12px 24px;
             box-shadow: 0 4px 14px rgba(79, 70, 229, .3); transition: background .15s; }
  .primary:hover { background: var(--brand-dark); }
  .primary:disabled { opacity: .6; cursor: progress; }

  .error { margin-top: 20px; padding: 14px 16px; border-radius: 12px; white-space: pre-line;
           color: #991b1b; background: var(--high-bg); border: 1px solid var(--high-line); }

  .result { margin-top: 20px; padding: 26px; border-radius: var(--radius); box-shadow: var(--shadow);
            background: var(--c-bg); border: 1px solid var(--c-line); border-top: 6px solid var(--c); }
  .result.Low    { --c: var(--low);  --c-bg: var(--low-bg);  --c-line: var(--low-line); }
  .result.Medium { --c: var(--med);  --c-bg: var(--med-bg);  --c-line: var(--med-line); }
  .result.High   { --c: var(--high); --c-bg: var(--high-bg); --c-line: var(--high-line); }
  .top { display: flex; flex-wrap: wrap; align-items: center; gap: 18px 22px; }
  .score { font-size: 68px; font-weight: 800; line-height: 1; letter-spacing: -.04em; color: var(--c);
           font-variant-numeric: tabular-nums; }
  .score small { font-size: 20px; font-weight: 600; letter-spacing: 0; color: var(--muted); }
  .badge { color: #fff; background: var(--c); font-size: 14px; font-weight: 700; letter-spacing: .06em;
           text-transform: uppercase; padding: 6px 14px; border-radius: 999px; }
  .meter { flex: 1 1 220px; height: 10px; border-radius: 999px; overflow: hidden;
           background: rgba(15, 23, 42, .08); }
  .meter div { height: 100%; width: 0; border-radius: 999px; background: var(--c); transition: width .6s ease; }
  .explain { margin: 18px 0 14px; font-size: 16px; }
  .tags { display: flex; flex-wrap: wrap; gap: 8px; }
  .tag { font-size: 12px; font-weight: 600; color: var(--c); background: #fff;
         border: 1px solid var(--c-line); padding: 4px 10px; border-radius: 999px; }
  .meta { margin: 14px 0 0; color: var(--muted); font-size: 13px; }

  @media (max-width: 640px) {
    .grid { grid-template-columns: 1fr; }
    .check { padding-top: 0; }
    .score { font-size: 54px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div>
      <h1>Fraud Risk Scoring</h1>
      <p class="sub">Isolation Forest + weighted risk rules, served by FastAPI</p>
    </div>
    <nav class="links">
      <span id="health" class="pill">Checking model&hellip;</span>
      <a href="/docs">API docs</a>
      <a href="/api">API info</a>
    </nav>
  </header>

  <div class="card">
    <div class="presets">
      <span>Quick fill:</span>
      <button type="button" class="preset" style="--dot: var(--low)"  data-preset="low">Low Risk Example</button>
      <button type="button" class="preset" style="--dot: var(--med)"  data-preset="medium">Medium Risk Example</button>
      <button type="button" class="preset" style="--dot: var(--high)" data-preset="high">High Risk Example</button>
    </div>

    <form id="order-form">
      <div class="grid">
        <div><label for="customer_id">Customer ID</label>
             <input id="customer_id" required placeholder="CUST-000123"></div>
        <div><label for="order_amount">Order amount (INR)</label>
             <input id="order_amount" type="number" min="1" step="0.01" required></div>
        <div><label for="product_category">Product category</label>
             <select id="product_category" required>
               <option value="electronics">Electronics</option>
               <option value="clothing">Clothing</option>
               <option value="groceries">Groceries</option>
               <option value="appliances">Appliances</option>
               <!-- the model knows this category as "books_media" -->
               <option value="books_media">Books</option>
             </select></div>
        <div><label for="quantity">Quantity</label>
             <input id="quantity" type="number" min="1" step="1" value="1" required></div>
        <div><label for="timestamp">Order timestamp</label>
             <input id="timestamp" type="datetime-local" required></div>
        <div><label for="account_creation_date">Account created</label>
             <input id="account_creation_date" type="datetime-local" required></div>
        <div><label for="shipping_address_id">Shipping address ID</label>
             <input id="shipping_address_id" required placeholder="ADDR-000123"></div>
        <div><label for="billing_address_id">Billing address ID</label>
             <input id="billing_address_id" required placeholder="ADDR-000123"></div>
        <div><label for="device_id">Device ID</label>
             <input id="device_id" required placeholder="DEV-000123"></div>
        <div><label for="ip_address">IP address</label>
             <input id="ip_address" required placeholder="117.196.44.12"></div>
        <div class="check"><input id="is_returned" type="checkbox">
             <label for="is_returned">Order was returned</label></div>
      </div>

      <details id="lookups">
        <summary>History lookups (optional)</summary>
        <p class="hint">A single order can't reveal these. In production the backend looks them up
          in the orders database. Leave them blank to treat this customer, device and address
          as brand new.</p>
        <div class="grid">
          <div><label for="customer_order_count">Customer's total orders</label>
               <input id="customer_order_count" type="number" min="0" step="1"></div>
          <div><label for="customer_return_rate">Customer return rate (0&ndash;1)</label>
               <input id="customer_return_rate" type="number" min="0" max="1" step="0.01"></div>
          <div><label for="orders_last_24h">Orders in the previous 24h</label>
               <input id="orders_last_24h" type="number" min="0" step="1"></div>
          <div><label for="customer_avg_amount">Customer's average order (INR)</label>
               <input id="customer_avg_amount" type="number" min="0" step="0.01"></div>
          <div><label for="device_reuse_count">Accounts on this device</label>
               <input id="device_reuse_count" type="number" min="0" step="1"></div>
          <div><label for="address_reuse_count">Accounts on this shipping address</label>
               <input id="address_reuse_count" type="number" min="0" step="1"></div>
          <div><label for="ip_reuse_count">Accounts on this IP</label>
               <input id="ip_reuse_count" type="number" min="0" step="1"></div>
          <div><label for="address_device_pair_count">Accounts on this address + device</label>
               <input id="address_device_pair_count" type="number" min="0" step="1"></div>
          <div><label for="cluster_signup_window_days">Signup window of those accounts (days)</label>
               <input id="cluster_signup_window_days" type="number" min="0" step="1"></div>
        </div>
      </details>

      <div class="actions">
        <button id="submit" class="primary" type="submit">Score This Order</button>
      </div>
    </form>
  </div>

  <div id="error" class="error" role="alert" hidden></div>

  <div id="result" class="result" aria-live="polite" hidden>
    <div class="top">
      <div class="score"><span id="r-score">0</span><small>/100</small></div>
      <span id="r-badge" class="badge"></span>
      <div class="meter"><div id="r-meter"></div></div>
    </div>
    <p id="r-explain" class="explain"></p>
    <div id="r-tags" class="tags"></div>
    <p id="r-meta" class="meta"></p>
  </div>

</div>

<script>
// ---- Preset orders for the quick-fill buttons ----------------------------------------
// Keys match the form field ids, which match the JSON field names the API expects.
const PRESETS = {
  low: {   // loyal 4-year-old account, ordinary clothing order, own address and device
    customer_id: "CUST-900001", order_amount: 1799, product_category: "clothing", quantity: 1,
    timestamp: "2026-09-11T19:45", account_creation_date: "2022-06-15T10:30",
    shipping_address_id: "ADDR-900001", billing_address_id: "ADDR-900001",
    device_id: "DEV-900001", ip_address: "117.196.44.12", is_returned: false,
    customer_order_count: 14, customer_return_rate: 0.07, orders_last_24h: 0,
    customer_avg_amount: 2100, device_reuse_count: 1, address_reuse_count: 1,
    ip_reuse_count: 1, address_device_pair_count: 1,
  },
  medium: { // established account, normal grocery basket, but on a device 6 accounts use
    customer_id: "CUST-900003", order_amount: 1450, product_category: "groceries", quantity: 6,
    timestamp: "2026-09-11T14:10", account_creation_date: "2025-07-20T09:00",
    shipping_address_id: "ADDR-900003", billing_address_id: "ADDR-900003",
    device_id: "DEV-900003", ip_address: "103.87.12.201", is_returned: false,
    customer_order_count: 6, customer_return_rate: 0, orders_last_24h: 0,
    customer_avg_amount: 1300, device_reuse_count: 6, address_reuse_count: 1,
    ip_reuse_count: 1, address_device_pair_count: 1,
  },
  high: {  // 1-day-old account, 3x-average electronics at 2:40 AM, fraud-ring device + address
    customer_id: "CUST-900002", order_amount: 94999, product_category: "electronics", quantity: 1,
    timestamp: "2026-09-11T02:40", account_creation_date: "2026-09-10T23:15",
    shipping_address_id: "ADDR-900002", billing_address_id: "ADDR-900099",
    device_id: "DEV-900002", ip_address: "157.32.76.84", is_returned: false,
    customer_order_count: 1, orders_last_24h: 0,
    device_reuse_count: 7, address_reuse_count: 6, ip_reuse_count: 7,
    address_device_pair_count: 6, cluster_signup_window_days: 4,
  },
};

// The optional history-lookup fields. Blank ones are left out of the request,
// so the API applies its "brand new customer" defaults.
const LOOKUPS = ["customer_order_count", "customer_return_rate", "orders_last_24h",
  "customer_avg_amount", "device_reuse_count", "address_reuse_count", "ip_reuse_count",
  "address_device_pair_count", "cluster_signup_window_days"];

const $ = (id) => document.getElementById(id);

function fillForm(preset) {
  for (const [key, value] of Object.entries(preset)) {
    const el = $(key);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value;
  }
  for (const key of LOOKUPS) if (!(key in preset)) $(key).value = "";
  $("lookups").open = true;             // show the lookups - they explain the score
  $("result").hidden = true;
  $("error").hidden = true;
}

// Turn the form into the JSON body POST /score-order expects.
function buildPayload() {
  const ts = $("timestamp").value;       // datetime-local gives "2026-09-11T14:10"
  const payload = {
    customer_id: $("customer_id").value.trim(),
    order_amount: Number($("order_amount").value),
    product_category: $("product_category").value,
    timestamp: ts.length === 16 ? ts + ":00" : ts,
    // The API field is a date, so keep only "YYYY-MM-DD" from the datetime-local value.
    account_creation_date: $("account_creation_date").value.slice(0, 10),
    shipping_address_id: $("shipping_address_id").value.trim(),
    billing_address_id: $("billing_address_id").value.trim(),
    device_id: $("device_id").value.trim(),
    ip_address: $("ip_address").value.trim(),
    quantity: Number($("quantity").value),
    is_returned: $("is_returned").checked ? 1 : 0,
  };
  for (const key of LOOKUPS) {
    const raw = $(key).value;
    if (raw !== "") payload[key] = Number(raw);
  }
  return payload;
}

// Turn an error response into a readable message.
function describeError(body) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) {           // 422 from pydantic: one entry per bad field
    return "Please fix the following:\n" + detail.map((e) => {
      const field = (e.loc || []).filter((part) => part !== "body").join(".") || "request";
      return "• " + field + ": " + e.msg;
    }).join("\n");
  }
  if (typeof detail === "string") return detail;   // HTTPException raised by our own code
  return "The server returned an unexpected response.";
}

function showError(message) {
  const box = $("error");
  box.textContent = message;             // textContent, never innerHTML: no injected markup
  box.hidden = false;
  $("result").hidden = true;
}

const prettyFlag = (flag) => flag.replace(/_flag$/, "").replace(/_/g, " ");

function addTag(container, text, title) {
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = text;
  if (title) tag.title = title;
  container.appendChild(tag);
}

function showResult(r) {
  const card = $("result");
  card.className = "result " + r.risk_bucket;          // Low / Medium / High -> colour
  $("r-score").textContent = Number.isInteger(r.risk_score) ? r.risk_score : r.risk_score.toFixed(1);
  $("r-badge").textContent = r.risk_bucket + " risk";
  $("r-explain").textContent = r.explanation;

  const tags = $("r-tags");
  tags.replaceChildren();
  const flags = r.flags_triggered || [];
  for (const flag of flags) addTag(tags, prettyFlag(flag), flag);
  if (!flags.length) addTag(tags, "no rules fired");

  $("r-meta").textContent = "Anomaly score " + r.anomaly_score.toFixed(4) + "  ·  "
    + flags.length + (flags.length === 1 ? " rule" : " rules") + " triggered";

  $("error").hidden = true;
  card.hidden = false;
  const meter = $("r-meter");
  meter.style.width = "0";
  requestAnimationFrame(() => { meter.style.width = Math.min(100, r.risk_score) + "%"; });
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---- Submit: call the API ------------------------------------------------------------
$("order-form").addEventListener("submit", async (event) => {
  event.preventDefault();                // stop the browser from reloading the page
  const button = $("submit");
  button.disabled = true;
  button.textContent = "Scoring…";

  try {
    // Same server that served this page, so a relative URL is all we need.
    const response = await fetch("/score-order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    let body = null;
    try { body = await response.json(); } catch (_) { /* not JSON - handled below */ }

    if (!response.ok) {
      showError("Error " + response.status + ": " + describeError(body));
    } else {
      showResult(body);
    }
  } catch (err) {
    // fetch() only throws when no response arrived at all (server down, network error).
    showError("Could not reach the API - is the server still running?\n(" + err.message + ")");
  } finally {
    button.disabled = false;
    button.textContent = "Score This Order";
  }
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => fillForm(PRESETS[button.dataset.preset]));
});

// ---- Header pill: is the model loaded? -------------------------------------------------
fetch("/health")
  .then(async (response) => {
    const body = await response.json();
    const pill = $("health");
    if (response.ok && body.model_loaded) {
      pill.textContent = "Model loaded · v" + body.model_version;
      pill.className = "pill ok";
    } else {
      pill.textContent = "Model not loaded";
      pill.className = "pill bad";
      pill.title = body.error || "";
    }
  })
  .catch(() => {
    $("health").textContent = "API unreachable";
    $("health").className = "pill bad";
  });

fillForm(PRESETS.low);                   // start with a valid example in the form
$("lookups").open = false;
</script>
</body>
</html>
"""