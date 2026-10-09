import json
from pathlib import Path

from diagnose_envelope_xjtu_table2 import diagnose_sample, format_harmonics


RESULT_DIR = Path("test_results_windowed_ms2ae")
OUT_JSON = RESULT_DIR / "xjtu_user_table_range_comparison.json"
OUT_MD = RESULT_DIR / "xjtu_user_table_range_comparison.md"

ROWS = [
    {
        "dataset": "XJTU2-1",
        "range": "#450-#457",
        "sample": 453,
        "stage": "Medium",
        "expected_faults": {"BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X, 2X BPFI - 1X BSF - 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU2-1",
        "range": "#455-#460",
        "sample": 457,
        "stage": "Medium",
        "expected_faults": {"BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X, 2X BPFI - 1X BSF - 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU2-1",
        "range": "#460-#470",
        "sample": 465,
        "stage": "Last",
        "expected_faults": {"BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X BPFI - 1X BSF - 2X, 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU2-3",
        "range": "#301-#311",
        "sample": 306,
        "stage": "Early",
        "expected_faults": {"BSF", "FTF"},
        "expected_diagnosis": "1X BSF - 1X, 2X, 4X, 5X, 6X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU2-3",
        "range": "#410-#420",
        "sample": 415,
        "stage": "Medium",
        "expected_faults": {"BPFO", "BPFI", "FTF"},
        "expected_diagnosis": "1X BPFO - 1X BPFI - 1X, 2X, 4X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU2-3",
        "range": "#525-#532",
        "sample": 528,
        "stage": "Medium",
        "expected_faults": {"BPFO", "BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X, 5X, 6X BPFO - 2X BPFI - 1X, 3X BSF - 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    },
    {
        "dataset": "XJTU3-1",
        "range": "#2340-#2360",
        "sample": 2350,
        "stage": "Early",
        "expected_faults": {"BPFO", "BSF", "FTF"},
        "expected_diagnosis": "1X, 2X BPFO - 1X, 3X BSF - 3X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
    {
        "dataset": "XJTU3-1",
        "range": "#2440-#2460",
        "sample": 2450,
        "stage": "Medium",
        "expected_faults": {"BPFO", "BPFI", "BSF", "FTF"},
        "expected_diagnosis": "2X, 3X BPFO - 1X BPFI - 1X, 3X BSF - 1X, 3X, 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
    {
        "dataset": "XJTU3-1",
        "range": "#2520-#2540",
        "sample": 2530,
        "stage": "Last",
        "expected_faults": {"BPFO", "BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X, 2X, 3X, 5X BPFO - 5X BPFI - 2X, 3X BSF - 6X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
    {
        "dataset": "XJTU3-4",
        "range": "#1410-#1425",
        "sample": 1417,
        "stage": "Early",
        "expected_faults": {"BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X BPFI - 1X BSF - 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
    {
        "dataset": "XJTU3-4",
        "range": "#1445-#1460",
        "sample": 1450,
        "stage": "Medium",
        "expected_faults": {"BPFO", "BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X BPFO - 1X BPFI - 1X, 3X BSF - 2X, 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
    {
        "dataset": "XJTU3-4",
        "range": "#1495-#1510",
        "sample": 1502,
        "stage": "Last",
        "expected_faults": {"BPFO", "BPFI", "BSF", "FTF"},
        "expected_diagnosis": "1X BPFO - 1X, 2X BPFI - 1X BSF - 2X, 5X FTF",
        "meta": {"fs": 25600.0, "BPFO": 123.20, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    },
]


def project_faults(diagnosis):
    return set(diagnosis["detected_fault_frequencies"])


def status(expected, actual):
    if expected == actual:
        return "MATCH"
    if expected and expected.issubset(actual):
        return "OVER-DETECTED"
    if actual and actual.issubset(expected):
        return "PARTIAL"
    if expected & actual:
        return "MIXED"
    return "MISS"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for row in ROWS:
        print(f"[COMPARE] {row['dataset']} {row['range']} sample={row['sample']}", flush=True)
        diagnosis = diagnose_sample(row["dataset"], row["sample"], row["meta"])
        expected = row["expected_faults"]
        actual = project_faults(diagnosis)
        results.append({
            "dataset": row["dataset"],
            "range": row["range"],
            "representative_sample": row["sample"],
            "stage": row["stage"],
            "expected_faults": sorted(expected),
            "project_faults": sorted(actual),
            "status": status(expected, actual),
            "expected_diagnosis": row["expected_diagnosis"],
            "project_diagnosis": format_harmonics(diagnosis["harmonics"]),
            "bandpass_range": diagnosis["bandpass_range"],
        })

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# XJTU User Table Comparison",
        "",
        "Each row uses one representative sample at the center of the range.",
        "",
        "| Dataset | Range | Stage | Expected faults | Project faults | Status | Project diagnosis |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['range']} | {row['stage']} | "
            f"{', '.join(row['expected_faults'])} | {', '.join(row['project_faults']) or 'None'} | "
            f"{row['status']} | {row['project_diagnosis']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {OUT_JSON}", flush=True)
    print(f"[SAVED] {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
