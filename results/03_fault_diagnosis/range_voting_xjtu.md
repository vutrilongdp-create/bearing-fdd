# XJTU Range Voting Comparison

Rules: top 10 peaks, amplitude >= 5% of max, harmonic tolerance ±5.0 Hz, fault vote >= 30% samples.

| Dataset | Range | Stage | Expected faults | Project faults | Status | Votes | Project diagnosis |
|---|---|---|---|---|---|---|---|
| XJTU2-1 | #450-#457 | Medium | BPFI, BSF, FTF | BSF, FTF | PARTIAL | {'BPFI': 1, 'BSF': 7, 'FTF': 8, 'BPFO': 1} | 1X BSF - 4X, 5X, 2X, 1X FTF |
| XJTU2-1 | #455-#460 | Medium | BPFI, BSF, FTF | BSF, FTF | PARTIAL | {'BPFO': 1, 'BSF': 5, 'FTF': 6} | 1X, 2X BSF - 1X, 5X, 4X, 2X FTF |
| XJTU2-1 | #460-#470 | Last | BPFI, BSF, FTF | BPFI, BSF, FTF | MATCH | {'BSF': 11, 'FTF': 11, 'BPFI': 10, 'BPFO': 2} | 1X, 2X BPFI - 1X BSF - 5X FTF |
| XJTU2-3 | #301-#311 | Early | BSF, FTF | BPFO, BSF, FTF | OVER-DETECTED | {'BPFO': 6, 'BSF': 11, 'FTF': 11} | 1X BPFO - 1X BSF - 2X, 4X, 5X, 1X, 6X FTF |
| XJTU2-3 | #410-#420 | Medium | BPFI, BPFO, FTF | BPFO, BSF, FTF | MIXED | {'BPFO': 11, 'BSF': 10, 'FTF': 11} | 1X BPFO - 1X BSF - 5X, 2X, 3X, 4X FTF |
| XJTU2-3 | #525-#532 | Medium | BPFI, BPFO, BSF, FTF | BPFI, BPFO, BSF, FTF | MATCH | {'BPFO': 7, 'BSF': 5, 'FTF': 6, 'BPFI': 3} | 1X BPFO - 1X BSF - 2X, 5X FTF |
| XJTU3-1 | #2340-#2360 | Early | BPFO, BSF, FTF | BPFI, BPFO, BSF, FTF | OVER-DETECTED | {'BSF': 13, 'FTF': 20, 'BPFI': 8, 'BPFO': 10} | 1X BPFO - 1X BSF - 1X FTF |
| XJTU3-1 | #2440-#2460 | Medium | BPFI, BPFO, BSF, FTF | BPFI, BPFO, BSF, FTF | MATCH | {'BPFI': 8, 'BSF': 13, 'FTF': 20, 'BPFO': 13} | 1X BPFO - 1X, 3X, 2X FTF |
| XJTU3-1 | #2520-#2540 | Last | BPFI, BPFO, BSF, FTF | BPFI, BPFO, BSF, FTF | MATCH | {'BPFI': 7, 'FTF': 12, 'BPFO': 9, 'BSF': 8} | 1X BPFO - 2X BSF - 1X, 4X, 2X FTF |
| XJTU3-4 | #1410-#1425 | Early | BPFI, BSF, FTF | BPFI, BPFO, BSF, FTF | OVER-DETECTED | {'BPFI': 9, 'BPFO': 9, 'FTF': 12, 'BSF': 9} | 1X BPFO - 1X BPFI - 2X BSF - 5X, 2X FTF |
| XJTU3-4 | #1445-#1460 | Medium | BPFI, BPFO, BSF, FTF | BPFI, BPFO, BSF, FTF | MATCH | {'BPFI': 6, 'FTF': 15, 'BSF': 11, 'BPFO': 10} | 1X BPFO - 1X BSF - 4X, 3X, 2X FTF |
| XJTU3-4 | #1495-#1510 | Last | BPFI, BPFO, BSF, FTF | BPFO, BSF, FTF | PARTIAL | {'BPFI': 4, 'BPFO': 5, 'BSF': 9, 'FTF': 16} | 1X BSF - 5X, 2X FTF |
