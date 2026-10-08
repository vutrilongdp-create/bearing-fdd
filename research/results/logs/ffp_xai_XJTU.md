# Project FFP/Threshold/XAI Refresh

- Aggregation: `recon_p95`
- Healthy samples: `300`
- FFP rule: first `5` consecutive samples over P95 threshold

| Dataset | Project FFP | Paper FFP | Diff | Threshold | Top 3 correlated features |
|---|---:|---:|---:|---:|---|
| XJTU2-1 | 383 | 451 | -68 | 0.185457 | BSF f (0.93); RMS (0.93); BPFI nf (0.86) |
| XJTU2-3 | 301 | 301 | 0 | 0.378661 | RMS (0.98); BPFO nf (0.91); Fund nf (0.82) |
| XJTU3-1 | 1268 | 2347 | -1079 | 0.281536 | BPFO nf (0.91); RMS (0.90); BPFI nf (0.87) |
| XJTU3-4 | 468 | 1416 | -948 | 0.311832 | RMS (0.98); BPFI nf (0.91); BSF f (0.91) |
