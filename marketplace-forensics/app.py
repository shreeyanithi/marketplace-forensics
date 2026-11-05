import json
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from pipeline import OUT, run

app = Flask(__name__, static_folder="static")
if not (OUT / "sellers.json").exists():
    run()
SELLERS = json.loads((OUT / "sellers.json").read_text())
META = json.loads((OUT / "meta.json").read_text())
DAILY = {k: g for k, g in pd.read_csv(OUT / "daily.csv").groupby("seller_id")}
INDEX = {s["seller_id"]: s for s in SELLERS}


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/summary")
def summary():
    return jsonify(META)


@app.route("/api/sellers")
def sellers():
    p, q = request.args.get("priority"), request.args.get("q", "").lower()
    rows = [s for s in SELLERS if (not p or s["priority"] == p) and q in s["seller_id"].lower()]
    keys = ["seller_id", "state", "orders", "revenue", "avg_rating", "score", "priority"]
    return jsonify([{k: s[k] for k in keys} for s in rows[:200]])


@app.route("/api/seller/<sid>")
def seller(sid):
    if sid not in INDEX:
        return jsonify(error="not found"), 404
    g = DAILY[sid]
    return jsonify(**INDEX[sid], daily=g.to_dict("list"))


if __name__ == "__main__":
    app.run(debug=True)
