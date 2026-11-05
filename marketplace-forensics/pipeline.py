"""Forensic analytics pipeline: Olist CSVs -> per-seller evidence profiles."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
D, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def load():
    rd = lambda n, **k: pd.read_csv(D / f"olist_{n}_dataset.csv", **k)
    o = rd("orders", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                  "order_estimated_delivery_date"])
    return o, rd("order_items"), rd("order_reviews"), rd("customers"), rd("sellers")


def z(s):
    s = s.fillna(s.median())
    return (s - s.mean()) / (s.std(ddof=0) or 1)


def iqr_out(s):
    s = s.fillna(s.median())
    q1, q3 = s.quantile([.25, .75])
    return (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))


def benford(prices):
    p = prices[prices > 0].to_numpy()
    if len(p) < 100:
        return None
    d = np.floor(p / 10 ** np.floor(np.log10(p))).astype(int).clip(1, 9)
    obs = np.bincount(d, minlength=10)[1:] / len(d)
    return {"observed": obs.round(4).tolist(), "mad": float(np.abs(obs - BENFORD).mean()),
            "p": float(chisquare(obs * len(d), BENFORD * len(d)).pvalue), "n": len(d)}


def run():
    o, it, rv, cu, se = load()
    # ---- seller-order level table (preprocessing from FORENSEIC.ipynb) ----
    so = (it.groupby(["seller_id", "order_id"]).agg(value=("price", "sum")).reset_index()
          .merge(o, on="order_id").merge(cu[["customer_id", "customer_unique_id"]], on="customer_id"))
    so["late"] = (so.order_delivered_customer_date > so.order_estimated_delivery_date).astype(int)
    so["cancel"] = (so.order_status == "canceled").astype(int)
    so["date"] = so.order_purchase_timestamp.dt.normalize()

    f = so.groupby("seller_id").agg(orders=("order_id", "nunique"), revenue=("value", "sum"),
                                    customers=("customer_unique_id", "nunique"), aov=("value", "mean"),
                                    late_rate=("late", "mean"), cancel_rate=("cancel", "mean")).reset_index()
    f["repeat_ratio"] = f.orders / f.customers
    f = f.merge(it.groupby("seller_id").agg(avg_price=("price", "mean"), products=("product_id", "nunique")).reset_index())
    f = f.merge(se[["seller_id", "seller_state"]], how="left")

    # reviews: single-seller orders only, so ratings are attributable
    n = it.groupby("order_id").seller_id.nunique()
    sr = so[so.order_id.isin(n[n == 1].index)].merge(
        rv.groupby("order_id").agg(score=("review_score", "mean")).reset_index(), on="order_id")
    sr["five"] = (sr.score == 5).astype(int)
    f = f.merge(sr.groupby("seller_id").agg(reviews=("score", "size"), rating=("score", "mean"),
                                            five_ratio=("five", "mean")).reset_index(), how="left")

    # ---- time series: daily orders / 5-star reviews, onset & peak ----
    dd = so.groupby(["seller_id", "date"]).agg(orders=("order_id", "nunique")).reset_index()
    dd = dd.merge(sr.groupby(["seller_id", "date"]).five.sum().rename("five_star").reset_index(), how="left")
    dd["five_star"] = dd.five_star.fillna(0).astype(int)
    ts = {}
    for sid, g in dd.groupby("seller_id"):
        m, s = g.orders.mean(), g.orders.std(ddof=0)
        hot = g[g.orders > max(m + 3 * s, 3)]
        ts[sid] = dict(peak=g.loc[g.orders.idxmax(), "date"], peak_orders=int(g.orders.max()),
                       onset=hot.date.min() if len(hot) else None,
                       burst=float(g.orders.max() / m), five_burst=float(g.five_star.max() / max(g.five_star.mean(), 1e-9)))
    t = pd.DataFrame.from_dict(ts, orient="index").rename_axis("seller_id").reset_index()
    f = f.merge(t, on="seller_id", how="left")

    # ---- statistical layer ----
    f["vol_z"] = pd.concat([z(f.orders).abs(), z(f.revenue).abs(), z(f.customers).abs()], axis=1).max(axis=1)
    f["vol_iqr"] = sum(iqr_out(f[c]).astype(int) for c in ["orders", "revenue", "customers", "avg_price"])
    b, a = np.polyfit(np.log1p(f.orders), np.log1p(f.revenue), 1)          # regression: revenue ~ orders
    res = np.log1p(f.revenue) - (a + b * np.log1p(f.orders))
    f["reg_z"] = (res / res.std()).abs()
    cols = ["orders", "revenue", "customers", "aov", "avg_price", "products", "late_rate", "cancel_rate",
            "reviews", "rating", "five_ratio", "repeat_ratio", "peak_orders"]
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(f[cols]))
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(X)
    f["iso_score"], f["iso_flag"] = iso.decision_function(X), iso.predict(X) == -1
    corr = f[cols].corr(numeric_only=True).round(2)

    # ---- evidence profiles ----
    recs, d = [], lambda x: None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")
    bf_all = it.groupby("seller_id").price.apply(benford)
    for _, r in f.iterrows():
        bf, sig = bf_all.get(r.seller_id), []
        add = lambda k, on, txt, w: sig.append(dict(name=k, triggered=bool(on), detail=txt, weight=w))
        add("Volume outlier", r.vol_z >= 3 or r.vol_iqr >= 3, f"max |z|={r.vol_z:.1f}, IQR flags={r.vol_iqr}/4", 1)
        add("Multivariate anomaly (Isolation Forest)", r.iso_flag, f"score={r.iso_score:.3f}", 1.5)
        add("Benford's Law deviation", bf and bf["mad"] > 0.015 and bf["p"] < 0.01,
            f"first-digit MAD={bf['mad']:.3f} (n={bf['n']})" if bf else "too few items (<100)", 1)
        add("Review burst", r.orders >= 20 and r.five_burst >= 6 and (r.five_ratio or 0) >= 0.9,
            f"5-star peak {r.five_burst:.1f}x daily mean, 5-star share {r.five_ratio or 0:.0%}", 2)
        add("Revenue/volume regression residual", r.reg_z >= 3, f"|resid z|={r.reg_z:.1f}", 1)
        add("Repeat-customer concentration", r.orders >= 10 and r.repeat_ratio > 1.3,
            f"{r.repeat_ratio:.2f} orders per unique customer", 2)
        add("Delivery / cancellation abuse", r.orders >= 10 and (r.late_rate > .3 or r.cancel_rate > .1),
            f"late {r.late_rate:.0%}, cancelled {r.cancel_rate:.0%}", 1)
        ex, pk = [], r.peak
        if pd.notna(pk) and (pd.Timestamp(pk).month == 12 or (pd.Timestamp(pk).month == 11 and pd.Timestamp(pk).day >= 15)):
            ex.append(("Peak falls in Black Friday / holiday window - seasonal demand is a plausible cause", -1))
        if r.orders >= 100 and r.late_rate < .15 and (r.rating or 0) >= 3.8 and r.repeat_ratio < 1.15:
            ex.append(("Large, diverse customer base with good ratings and on-time delivery - scale looks like an established seller", -1.5))
        score = max(0, sum(s["weight"] for s in sig if s["triggered"]) + sum(w for _, w in ex))
        recs.append(dict(seller_id=r.seller_id, state=r.seller_state, orders=int(r.orders), revenue=round(r.revenue, 2),
                         customers=int(r.customers), avg_rating=None if pd.isna(r.rating) else round(r.rating, 2),
                         late_rate=round(r.late_rate, 3), cancel_rate=round(r.cancel_rate, 3), score=round(score, 2),
                         priority="High" if score >= 4 else "Medium" if score >= 2 else "Low",
                         signals=sig, explanations=[e for e, _ in ex], onset=d(r.onset), peak=d(r.peak),
                         peak_orders=int(r.peak_orders), benford=bf))
    recs.sort(key=lambda x: -x["score"])
    meta = dict(sellers=len(recs), priority={p: sum(x["priority"] == p for x in recs) for p in ["High", "Medium", "Low"]},
                signal_counts={k: sum(any(s["name"] == k and s["triggered"] for s in x["signals"]) for x in recs)
                               for k in [s["name"] for s in recs[0]["signals"]]},
                regression={"slope": round(b, 3), "intercept": round(a, 3)}, correlation=corr.to_dict())
    (OUT / "sellers.json").write_text(json.dumps(recs))
    (OUT / "meta.json").write_text(json.dumps(meta))
    dd.assign(date=dd.date.dt.strftime("%Y-%m-%d")).to_csv(OUT / "daily.csv", index=False)
    print("Done:", meta["priority"])


if __name__ == "__main__":
    run()
