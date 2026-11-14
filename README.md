# Frozen model assets

Already included: `hawkes_selected_model.json`, `selected_features.csv`,
`edge_table_static_robust_preview.csv` (all from final node-level source release).

**Required before real inference:** `epicpp_deployment_bundle.joblib` (6.8 MB)
from the completed V2.1 node-level AIMOS run. It is not in this archive.

Do **not** substitute the job-level model or retrain.
Run `tools/export_model_from_aimos.py` on AIMOS to create a ZIP containing
this frozen checkpoint, then extract that ZIP into this models directory.
Only load joblib from trusted sources.
