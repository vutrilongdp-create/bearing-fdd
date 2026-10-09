# IMS Range Voting Comparison

Rules: top 10 peaks, amplitude >= 5% of max, harmonic tolerance ±5.0 Hz, fault vote >= 30% samples.

| Dataset | Range | Stage | Expected faults | Project faults | Status | Votes | Project diagnosis |
|---|---|---|---|---|---|---|---|
| IMS1 | #1850-#1880 | Early | BPFI, FTF | FTF | PARTIAL | {'FTF': 31, 'BPFO': 9, 'BSF': 8, 'BPFI': 7} | 1X, 2X, 5X, 3X, 4X FTF |
| IMS1 | #2135-#2145 | Medium | BPFI, FTF | BPFI, BPFO, FTF | OVER-DETECTED | {'FTF': 7, 'BPFO': 4, 'BPFI': 8, 'BSF': 1} | 1X BPFI - 2X, 4X FTF |
| IMS1 | #2150-#2156 | Last | BPFI, BPFO, FTF | BPFI, FTF | PARTIAL | {'BPFI': 5, 'FTF': 6} | 1X BPFI - 2X, 4X FTF |
| IMS2 | #530-#550 | Early | BPFO, FTF | BPFO, FTF | MATCH | {'FTF': 21, 'BPFO': 19} | 1X, 2X BPFO - 2X, 1X, 3X, 5X, 4X, 6X FTF |
| IMS2 | #865-#875 | Medium | BPFO | BPFO | MATCH | {'BPFO': 11} | 1X, 2X, 3X, 4X, 5X BPFO |
| IMS2 | #970-#980 | Last | BPFO | BPFO | MATCH | {'BPFO': 11, 'FTF': 3, 'BPFI': 3} | 1X, 2X, 3X, 4X, 5X BPFO |
| IMS3 | #5960-#5990 | Early | FTF | BPFO, FTF | OVER-DETECTED | {'FTF': 30, 'BPFI': 8, 'BPFO': 11, 'BSF': 2} | 1X BPFO - 2X, 1X FTF |
| IMS3 | #6170-#6190 | Medium | BPFO | BPFI, BPFO | OVER-DETECTED | {'BPFO': 21, 'FTF': 1, 'BPFI': 7} | 1X, 2X, 3X, 4X, 5X BPFO - 4X BPFI |
| IMS3 | #6314-#6324 | Last | BPFO, FTF | BPFI, BPFO, FTF | OVER-DETECTED | {'BPFI': 4, 'BPFO': 9, 'FTF': 3} | 1X, 2X, 3X, 4X, 5X, 6X BPFO - 4X BPFI - 1X FTF |
