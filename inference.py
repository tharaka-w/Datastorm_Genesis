"""Inference-only V2.1 node-risk prediction with explanatory evidence.

Input is PREPARED unnormalized 15-minute node-frame telemetry and the EXACT
E5-compatible precursor event stream. We do not infer event types from raw
Nagios/Ganglia/IPMI files here.

No training, PCMCI+, Slurm submission, writes, or independent threshold tuning.
ECDF fusion is a ranking score, NOT a calibrated failure probability.
Feature-neutralization probes and PCMCI+ associations are NOT causal proof.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features_v21 import apply_baseline, build_features

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / 'models'
RAW_VARIABLES = (
    'pcie_power', 'gpu_mem_temp_max', 'gpu_mem_copy_util',
    'largest_job_nodes', 'node_ambient_temp', 'mem_cached'
)


def _hawkes(node: str, t: pd.Timestamp, history: pd.DataFrame | None, model: dict):
    """Match verified AIMOS E5 historical point-process score arithmetic."""
    if history is None:
        return None, None, []
    x = history.loc[history['node_id'].eq(node) & history['event_time'].le(t),
                    ['event_time', 'event_type']].copy()
    alpha = model['alpha_per_hour']
    x['alpha'] = x['event_type'].map(alpha).fillna(0).astype(float)
    x = x[x['alpha'] > 0]
    # Current-bin Nagios terminal events must not be used in current prediction.
    x = x[(x['event_type'] != 'NAGIOS_FAILURE') | x['event_time'].lt(t)]
    beta = float(model['beta_per_hour'])
    mu = float(model['mu_per_hour'])
    horizon = float(model.get('horizon_hours', 2.0))
    elapsed = (t - x['event_time']).dt.total_seconds().to_numpy() / 3600
    excitation = x['alpha'].to_numpy() * np.exp(-beta * elapsed)
    x['contribution'] = excitation
    q = float(np.sum(excitation))
    mass = horizon if beta == 0 else -math.expm1(-beta * horizon) / beta
    score = float(-math.expm1(-(mu * horizon + q * mass)))
    contrib = x.groupby('event_type')['contribution'].sum().sort_values(ascending=False)
    evidence = [{'event_type': str(k), 'excitation_per_hour': float(v)}
                for k, v in contrib.items() if v > 1e-10]
    return score, {'base_rate_per_hour': mu, 'event_excitation_per_hour': q,
                   'events_used': int(len(x)), 'horizon_hours': horizon}, evidence


def _require_model_files(artifacts_dir):
    base = Path(artifacts_dir).expanduser().resolve()
    bundle = base/'epicpp_deployment_bundle.joblib'
    hawkes = base/'hawkes_selected_model.json'
    missing = [str(p) for p in (bundle, hawkes) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Frozen model files missing. Run tools/export_model_from_aimos.py, '
                                'then extract ZIP into models/. Missing: '+', '.join(missing))
    return bundle, hawkes


def predict_and_explain(
    node_history: pd.DataFrame,
    hawkes_events: pd.DataFrame | None,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS,
    *,
    fused_alert_threshold: float | None = None,
    top_k: int = 8,
    require_complete_history: bool = True,
) -> list[dict]:
    """Score latest 15-minute time per node; return JSON-compatible dictionaries.

    node_history columns: node_id, timestamp, and six RAW variables in RAW_VARIABLES.
    Keep 9 bins per node covering t-120 through t at 15-minute spacing.
    The model itself uses prior bins, never future labels.

    hawkes_events columns: node_id, event_time, event_type, produced with the
    ORIGINAL E5 event definitions (z threshold etc.). None means unavailable;
    this triggers EXACT telemetry-only fusion fallback. An empty DataFrame means
    event history is complete but no events occurred (base Hawkes intensity).

    fused_alert_threshold is NOT provided by V2.1. Only supply a threshold
    chosen from legitimate development validation; otherwise alert is None.

    Load joblib files ONLY from a trusted source; they are Python pickles.
    No data or artifact files are modified by this function.
    """
    if not isinstance(node_history, pd.DataFrame) or node_history.empty:
        raise ValueError('node_history must be a nonempty pandas DataFrame')
    if top_k < 0:
        raise ValueError('top_k must be >= 0')
    if fused_alert_threshold is not None and not (0 <= float(fused_alert_threshold) <= 1):
        raise ValueError('fused_alert_threshold must be in [0,1]')

    bundle_path, hawkes_path = _require_model_files(artifacts_dir)
    b = joblib.load(bundle_path)  # TRUSTED bundle only.
    spec = b['selected_feature_spec']
    logical_vars = spec['feature_var'].astype(str).drop_duplicates().tolist()
    cols = list(b['feature_columns'])
    required = {'node_id','timestamp',*logical_vars}
    missing = required - set(node_history.columns)
    if missing:
        raise ValueError('Missing raw prepared node columns: '+', '.join(sorted(missing)))

    raw = node_history[['node_id','timestamp']+logical_vars].copy()
    if raw['node_id'].isna().any():
        raise ValueError('Null node_id')
    raw['node_id'] = raw['node_id'].astype(str).str.strip()
    if raw['node_id'].eq('').any():
        raise ValueError('Blank node_id')
    raw['timestamp'] = pd.to_datetime(raw['timestamp'], errors='coerce', utc=True)
    if raw['timestamp'].isna().any():
        raise ValueError('Invalid timestamp')
    if raw[['node_id','timestamp']].duplicated().any():
        raise ValueError('Duplicate node/timestamp rows: aggregate upstream')
    for v in logical_vars:
        raw[v] = pd.to_numeric(raw[v], errors='coerce')
    raw = raw.sort_values(['node_id','timestamp'],kind='mergesort').reset_index(drop=True)
    raw['month'] = raw['timestamp'].dt.strftime('%Y-%m')

    # Quality checks are per node before doing any model inference.
    quality = {}
    for node, sub in raw.groupby('node_id', sort=False):
        t = pd.Timestamp(sub['timestamp'].iloc[-1])
        bins = set(sub['timestamp'])
        expected = {t - pd.Timedelta(minutes=15*k) for k in range(9)}
        n = len(bins.intersection(expected))
        if require_complete_history and n != 9:
            raise ValueError(f'Node {node} at {t}: {n}/9 exact 15-min history bins; '
                             'set require_complete_history=False to score with a warning')
        quality[node] = {'consecutive_15min_history_bins':n, 'expected_history_bins':9}

    labels = raw[['node_id','timestamp']].copy()
    labels['failure_onset'] = 0
    labels['y'] = np.nan
    norm, _ = apply_baseline(raw, logical_vars, b['normalization_baseline'])
    feats, _ = build_features(norm, labels, spec)
    latest = (feats.sort_values(['node_id','timestamp'])
              .groupby('node_id',sort=False).tail(1))

    events = None
    model_h = None
    if hawkes_events is not None:
        if not isinstance(hawkes_events, pd.DataFrame):
            raise TypeError('hawkes_events must be a DataFrame or None')
        ecols = {'node_id','event_time','event_type'}
        if not ecols.issubset(hawkes_events.columns):
            raise ValueError('Missing E5 event columns: '+str(sorted(ecols-set(hawkes_events.columns))))
        events = hawkes_events[['node_id','event_time','event_type']].copy()
        if events['node_id'].isna().any():
            raise ValueError('Null E5 event node_id')
        events['node_id'] = events['node_id'].astype(str).str.strip()
        events['event_time'] = pd.to_datetime(events['event_time'],errors='coerce',utc=True)
        events = events.dropna(subset=['event_time','event_type'])
        model_h = json.loads(hawkes_path.read_text(encoding='utf-8'))

    model = b['telemetry_model']
    t_ref = np.asarray(b['telemetry_ecdf_reference_sorted'])
    h_ref = np.asarray(b['hawkes_ecdf_reference_sorted'])
    if not len(t_ref) or not len(h_ref):
        raise ValueError('Frozen ECDF reference arrays are empty')
    w = float(b['fusion']['hawkes_weight'])
    byvar = {}
    direct = {}
    for _, item in spec.iterrows():
        var = str(item['feature_var'])
        path = str(item.get('path',''))
        if path and path != 'nan':
            byvar.setdefault(var,[]).append(path)
            direct[str(item['feature_column'])] = path
    out = []
    for _, rec in latest.iterrows():
        node, timestamp = str(rec['node_id']), pd.Timestamp(rec['timestamp'])
        row = rec[cols].to_frame().T.astype(float)
        available = int(row.notna().sum(axis=1).iloc[0])
        if not available:
            raise ValueError(f'All HGB inputs are missing for {node} at {timestamp}')
        tel = float(model.predict_proba(row)[0,1])
        haw, haw_meta, hevidence = _hawkes(node,timestamp,events,model_h)
        tc = float(np.searchsorted(t_ref,tel,side='right')/len(t_ref))
        hc = (float(np.searchsorted(h_ref,haw,side='right')/len(h_ref))
              if haw is not None else tc)
        fused = float((1-w)*tc+w*hc)
        # Exactly one-feature-at-a-time neutralization; descriptive model sensitivity only.
        variants = pd.concat([row]*(len(cols)+1),ignore_index=True)
        for i,c in enumerate(cols,1):
            variants.at[i,c] = 0.0
        pr = model.predict_proba(variants[cols])[:,1]
        attrs = []
        for i,c in enumerate(cols):
            value = float(row.iloc[0][c]) if pd.notna(row.iloc[0][c]) else None
            key = c.split('__',1)[0]
            if c in direct:
                evidence_paths = [direct[c]]
                evidence_kind = 'exact_selected_lag'
            else:
                evidence_paths = list(dict.fromkeys(byvar.get(key, [])))[:3]
                evidence_kind = 'same_variable_family' if evidence_paths else 'none'
            attrs.append({
                'feature':c, 'hgb_score_delta_vs_zero':float(pr[0]-pr[i+1]),
                'normalized_feature_value':value,
                'pcmci_paths':evidence_paths,
                'path_match_kind':evidence_kind,
            })
        attrs.sort(key=lambda a:abs(a['hgb_score_delta_vs_zero']),reverse=True)
        dq = {
            'nonmissing_model_inputs':available,
            'total_model_inputs':len(cols),
            **quality[node],
            'history_complete':quality[node]['consecutive_15min_history_bins']==9,
        }
        if not dq['history_complete']:
            dq['warning'] = 'Irregular/missing 15-min history; score quality not validated'
        out.append({
            'node_id':node,'prediction_time':str(timestamp),
            'horizon_minutes':120,
            'risk_hgb_raw':tel,'risk_hawkes_raw':haw,
            'risk_telemetry_ecdf':tc,'risk_hawkes_ecdf_effective':hc,
            'risk_fused_rank':fused,
            'fused_alert':None if fused_alert_threshold is None else bool(fused>=fused_alert_threshold),
            'fused_alert_threshold':fused_alert_threshold,
            'hgb_only_alert_for_reference':bool(tel>=float(b['telemetry_threshold']))
                                            if b.get('telemetry_threshold') is not None else None,
            'data_quality':dq,
            'explanation':{
                'local_hgb_sensitivities':attrs[:top_k],
                'hawkes_event_contributions':hevidence,
                'hawkes_replay_details':haw_meta,
                'scope':('One-feature-zero sensitivity and learned PCMCI+ temporal '
                         'associations with n_critical_checks; not failure causation.'),
            },
            'interpretation':'The fused ECDF score is a risk rank, NOT a calibrated failure probability.',
        })
    return out
