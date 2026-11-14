"""Exact inference-time feature equations extracted from the verified V2.1 source.

NO fit, label construction, sampling, or retraining is exposed here.
The functions below were copied, without modifying their bodies, from
final_v2_1/run_global90_10.py in the recovered final source package.
"""
import gc
import numpy as np
import pandas as pd

CADENCE_MINUTES = 15
HORIZON_BINS = 8

def apply_baseline(raw, vars_, baseline):
    z = raw.copy()
    z["hour_utc"] = z["timestamp"].dt.hour.astype(np.int8)
    nh = baseline["node_hour"].rename(columns={v: f"__nh__{v}" for v in vars_})
    n = baseline["node"].rename(columns={v: f"__n__{v}" for v in vars_})
    h = baseline["hour"].rename(columns={v: f"__h__{v}" for v in vars_})
    g = baseline["global"]

    z = z.merge(nh, on=["node_id", "hour_utc"], how="left", validate="many_to_one")
    z = z.merge(n, on=["node_id"], how="left", validate="many_to_one")
    z = z.merge(h, on=["hour_utc"], how="left", validate="many_to_one")

    fallback = {}
    for v in vars_:
        b = z[f"__nh__{v}"]
        c1 = b.isna() & z[f"__n__{v}"].notna()
        b = b.fillna(z[f"__n__{v}"])
        c2 = b.isna() & z[f"__h__{v}"].notna()
        b = b.fillna(z[f"__h__{v}"])
        gv = g.get(v, np.nan)
        c3 = b.isna() & pd.notna(gv)
        b = b.fillna(gv)
        z[v] = (pd.to_numeric(z[v], errors="coerce") - b).astype(np.float32)
        fallback[v] = {
            "node_fallback": int(c1.sum()),
            "hour_fallback": int(c2.sum()),
            "global_fallback": int(c3.sum()),
            "still_missing": int(z[v].isna().sum()),
        }

    z = z.drop(columns=[c for c in z.columns if c.startswith("__")] + ["hour_utc"])
    return z, fallback


def exact_shift(z, grouped, var, lag):
    sv = grouped[var].shift(lag)
    st = grouped["timestamp"].shift(lag)
    ok = (z["timestamp"] - st).eq(pd.Timedelta(minutes=lag * CADENCE_MINUTES))
    return sv.where(ok).astype(np.float32), ok.to_numpy()


def build_features(norm, labels, spec):
    z = norm.merge(
        labels[["node_id", "timestamp", "failure_onset", "y"]],
        on=["node_id", "timestamp"], how="left", validate="one_to_one"
    )
    z = z.sort_values(["node_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = z.groupby("node_id", sort=False)
    out = z[["node_id", "timestamp", "month", "failure_onset", "y"]].copy()

    selected = []
    for _, r in spec.iterrows():
        var = str(r["feature_var"])
        lag = int(r["feature_lag_bins"])
        col = str(r["feature_column"])
        out[col] = exact_shift(z, grouped, var, lag)[0]
        selected.append(col)

    temporal = []
    logical_vars = spec["feature_var"].astype(str).drop_duplicates().tolist()
    xc = np.arange(HORIZON_BINS, dtype=np.float32)
    xc = xc - xc.mean()
    slope_denom = float(np.sum(xc * xc))

    for var in logical_vars:
        vals, masks = [], []
        for k in range(1, HORIZON_BINS + 1):
            s, ok = exact_shift(z, grouped, var, k)
            vals.append(s.to_numpy(dtype=np.float32, copy=False))
            masks.append(ok)

        mat = np.column_stack(vals).astype(np.float32, copy=False)
        mm = np.column_stack(masks)
        valid4 = mm[:, :4].all(axis=1)
        valid8 = mm.all(axis=1)
        valid_delta = mm[:, 0] & mm[:, 4]  # t-15 vs t-75 = 60 min apart

        feats = {}
        for name in ["past1h_mean", "past2h_mean", "past2h_std", "past2h_min",
                     "past2h_max", "past2h_range", "past2h_slope_per_hour", "delta_60m"]:
            feats[name] = np.full(len(z), np.nan, dtype=np.float32)

        if np.any(valid4):
            feats["past1h_mean"][valid4] = np.mean(mat[valid4, :4], axis=1).astype(np.float32)
        if np.any(valid8):
            m8 = mat[valid8]
            mn = np.min(m8, axis=1)
            mx = np.max(m8, axis=1)
            feats["past2h_mean"][valid8] = np.mean(m8, axis=1).astype(np.float32)
            feats["past2h_std"][valid8] = np.std(m8, axis=1).astype(np.float32)
            feats["past2h_min"][valid8] = mn.astype(np.float32)
            feats["past2h_max"][valid8] = mx.astype(np.float32)
            feats["past2h_range"][valid8] = (mx - mn).astype(np.float32)
            feats["past2h_slope_per_hour"][valid8] = ((m8 @ xc) / slope_denom * 4.0).astype(np.float32)
        if np.any(valid_delta):
            feats["delta_60m"][valid_delta] = (mat[valid_delta, 0] - mat[valid_delta, 4]).astype(np.float32)

        for name, arr in feats.items():
            c = f"{var}__{name}"
            out[c] = arr
            temporal.append(c)

        del mat, mm, vals, masks, feats
        gc.collect()

    hour = (out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0).to_numpy(dtype=np.float32)
    dow = out["timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    calendar = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    enhanced = selected + temporal + calendar
    out["all_selected_missing"] = out[selected].isna().all(axis=1)
    return out, {"selected": selected, "temporal": temporal, "calendar": calendar, "enhanced": enhanced}
