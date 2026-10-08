# Table 2 - Project Results Summary

Source: project rerun results in `ffp_xai_all.json`, `test_results_windowed_ms2ae/*_range_voting_comparison.json`, and the IMS-2 case-study summary.

| Dataset | Top 3 correlated features | Project FFP | Threshold | Sample/range | Stage | Bandpass filter range | Project diagnosis | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMS1 | RMS (0.94); BPFI f (0.90); BPFO f (0.87) | 1091 | 0.014475 | #1850 - #1880 | Early | 1280--2560 Hz | 1X, 2X, 5X, 3X, 4X FTF | PARTIAL |
| IMS1 | RMS (0.94); BPFI f (0.90); BPFO f (0.87) | 1091 | 0.014475 | #2135 - #2145 | Medium | 7680--8960 Hz | 1X BPFI - 2X, 4X FTF | OVER-DETECTED |
| IMS1 | RMS (0.94); BPFI f (0.90); BPFO f (0.87) | 1091 | 0.014475 | #2150 - #2156 | Last | 7680--8960 Hz | 1X BPFI - 2X, 4X FTF | PARTIAL |
| IMS2 | RMS (0.95); BPFO f (0.87); BSF nf (0.85) | 532 | 0.498045 | #530 - #550 | Early | 2560--5120 Hz | 1X, 2X BPFO - 2X, 1X, 3X, 5X, 4X, 6X FTF | MATCH |
| IMS2 | RMS (0.95); BPFO f (0.87); BSF nf (0.85) | 532 | 0.498045 | #865 - #875 | Medium | 5120--7680 Hz | 1X, 2X, 3X, 4X, 5X BPFO | MATCH |
| IMS2 | RMS (0.95); BPFO f (0.87); BSF nf (0.85) | 532 | 0.498045 | #970 - #980 | Last | 5120--7680 Hz | 1X, 2X, 3X, 4X, 5X BPFO | MATCH |
| IMS3 | RMS (0.93); BSF nf (0.90); FTF f (0.89) | 5940 | 0.005489 | #5960 - #5990 | Early | 2560--5120 Hz | 1X BPFO - 2X, 1X FTF | OVER-DETECTED |
| IMS3 | RMS (0.93); BSF nf (0.90); FTF f (0.89) | 5940 | 0.005489 | #6170 - #6190 | Medium | 2560--5120 Hz | 1X, 2X, 3X, 4X, 5X BPFO - 4X BPFI | OVER-DETECTED |
| IMS3 | RMS (0.93); BSF nf (0.90); FTF f (0.89) | 5940 | 0.005489 | #6314 - #6324 | Last | 5120--7680 Hz | 1X, 2X, 3X, 4X, 5X, 6X BPFO - 4X BPFI - 1X FTF | OVER-DETECTED |
| XJTU2-1 | BSF f (0.93); RMS (0.93); BPFI nf (0.86) | 383 | 0.185457 | #450 - #457 | Medium | 1600--3200 Hz | 1X BSF - 4X, 5X, 2X, 1X FTF | PARTIAL |
| XJTU2-1 | BSF f (0.93); RMS (0.93); BPFI nf (0.86) | 383 | 0.185457 | #455 - #460 | Medium | 1600--3200 Hz | 1X, 2X BSF - 1X, 5X, 4X, 2X FTF | PARTIAL |
| XJTU2-1 | BSF f (0.93); RMS (0.93); BPFI nf (0.86) | 383 | 0.185457 | #460 - #470 | Last | 1600--3200 Hz | 1X, 2X BPFI - 1X BSF - 5X FTF | MATCH |
| XJTU2-3 | RMS (0.98); BPFO nf (0.91); Fund nf (0.82) | 301 | 0.378661 | #301 - #311 | Early | 8533.33--10666.7 Hz | 1X BPFO - 1X BSF - 2X, 4X, 5X, 1X, 6X FTF | OVER-DETECTED |
| XJTU2-3 | RMS (0.98); BPFO nf (0.91); Fund nf (0.82) | 301 | 0.378661 | #410 - #420 | Medium | 8533.33--10666.7 Hz | 1X BPFO - 1X BSF - 5X, 2X, 3X, 4X FTF | MIXED |
| XJTU2-3 | RMS (0.98); BPFO nf (0.91); Fund nf (0.82) | 301 | 0.378661 | #525 - #532 | Medium | 9600--11200 Hz | 1X BPFO - 1X BSF - 2X, 5X FTF | MATCH |
| XJTU3-1 | BPFO nf (0.91); RMS (0.90); BPFI nf (0.87) | 1268 | 0.281536 | #2340 - #2360 | Early | 6400--8000 Hz | 1X BPFO - 1X BSF - 1X FTF | OVER-DETECTED |
| XJTU3-1 | BPFO nf (0.91); RMS (0.90); BPFI nf (0.87) | 1268 | 0.281536 | #2440 - #2460 | Medium | 6400--8000 Hz | 1X BPFO - 1X, 3X, 2X FTF | MATCH |
| XJTU3-1 | BPFO nf (0.91); RMS (0.90); BPFI nf (0.87) | 1268 | 0.281536 | #2520 - #2540 | Last | 6400--8533.33 Hz | 1X BPFO - 2X BSF - 1X, 4X, 2X FTF | MATCH |
| XJTU3-4 | RMS (0.98); BPFI nf (0.91); BSF f (0.91) | 468 | 0.311832 | #1410 - #1425 | Early | 3200--6400 Hz | 1X BPFO - 1X BPFI - 2X BSF - 5X, 2X FTF | OVER-DETECTED |
| XJTU3-4 | RMS (0.98); BPFI nf (0.91); BSF f (0.91) | 468 | 0.311832 | #1445 - #1460 | Medium | 4266.67--6400 Hz | 1X BPFO - 1X BSF - 4X, 3X, 2X FTF | MATCH |
| XJTU3-4 | RMS (0.98); BPFI nf (0.91); BSF f (0.91) | 468 | 0.311832 | #1495 - #1510 | Last | 3200--6400 Hz | 1X BSF - 5X, 2X FTF | PARTIAL |
