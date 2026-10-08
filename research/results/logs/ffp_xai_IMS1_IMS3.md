# Project FFP/Threshold/XAI Refresh

- Aggregation: `recon_p95`
- Healthy samples: `300`
- FFP rule: first `5` consecutive samples over P95 threshold

| Dataset | Project FFP | Paper FFP | Diff | Threshold | Top 3 correlated features |
|---|---:|---:|---:|---:|---|
| IMS1 | 1091 | 1857 | -766 | 0.014475 | -- |
| IMS3 | 5940 | 5967 | -27 | 0.005489 | RMS (0.93); BSF nf (0.90); FTF f (0.89) |
