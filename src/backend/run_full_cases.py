import json
import time
from pathlib import Path

import app as appmod


CASES = [
    {"group": "IMS", "label": "IMS1 early", "dataset": "IMS1", "healthy": 300, "first": 1850, "n": 30, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS1 medium", "dataset": "IMS1", "healthy": 300, "first": 2135, "n": 10, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS1 last", "dataset": "IMS1", "healthy": 300, "first": 2150, "n": 6, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS2 early", "dataset": "IMS2", "healthy": 300, "first": 530, "n": 20, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS2 medium", "dataset": "IMS2", "healthy": 300, "first": 865, "n": 10, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS2 last", "dataset": "IMS2", "healthy": 300, "first": 970, "n": 10, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS3 early", "dataset": "IMS3", "healthy": 300, "first": 5960, "n": 30, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS3 medium", "dataset": "IMS3", "healthy": 300, "first": 6170, "n": 20, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "IMS", "label": "IMS3 last", "dataset": "IMS3", "healthy": 300, "first": 6314, "n": 10, "fs": 20480, "shaft": 33, "BPFO": 236, "BPFI": 297, "BSF": 278, "FTF": 15},
    {"group": "XJTU-SY", "label": "XJTU2-1 medium A", "dataset": "XJTU2-1", "healthy": 300, "first": 450, "n": 7, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU2-1 medium B", "dataset": "XJTU2-1", "healthy": 300, "first": 455, "n": 5, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU2-1 last", "dataset": "XJTU2-1", "healthy": 300, "first": 460, "n": 10, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU2-3 early", "dataset": "XJTU2-3", "healthy": 300, "first": 301, "n": 10, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU2-3 medium A", "dataset": "XJTU2-3", "healthy": 300, "first": 410, "n": 10, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU2-3 medium B", "dataset": "XJTU2-3", "healthy": 300, "first": 525, "n": 7, "fs": 25600, "shaft": 37.5, "BPFO": 112.19, "BPFI": 178.94, "BSF": 75.21, "FTF": 14.20},
    {"group": "XJTU-SY", "label": "XJTU3-1 early", "dataset": "XJTU3-1", "healthy": 300, "first": 2340, "n": 20, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    {"group": "XJTU-SY", "label": "XJTU3-1 medium", "dataset": "XJTU3-1", "healthy": 300, "first": 2440, "n": 20, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    {"group": "XJTU-SY", "label": "XJTU3-1 last", "dataset": "XJTU3-1", "healthy": 300, "first": 2520, "n": 20, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    {"group": "XJTU-SY", "label": "XJTU3-4 early", "dataset": "XJTU3-4", "healthy": 300, "first": 1410, "n": 15, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    {"group": "XJTU-SY", "label": "XJTU3-4 medium", "dataset": "XJTU3-4", "healthy": 300, "first": 1445, "n": 15, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
    {"group": "XJTU-SY", "label": "XJTU3-4 last", "dataset": "XJTU3-4", "healthy": 300, "first": 1495, "n": 15, "fs": 25600, "shaft": 40, "BPFO": 123.2, "BPFI": 196.49, "BSF": 82.58, "FTF": 15.40},
]


def payload(case):
    return {
        "nombre_req": case["dataset"],
        "sampling_frequency_req": case["fs"],
        "bpfo_req": case["BPFO"],
        "bpfi_req": case["BPFI"],
        "bsf_req": case["BSF"],
        "ftf_req": case["FTF"],
        "shaft_frequency_req": case["shaft"],
        "healthy_number_req": case["healthy"],
        "analyzed_number_req": case["n"],
        "first_sample_req": case["first"],
    }


def run():
    client = appmod.app.test_client()
    results = []
    for index, case in enumerate(CASES, start=1):
        print(f"[{index:02d}/{len(CASES)}] {case['label']} ...", flush=True)
        start = time.time()
        response = client.post(f"/analyzeData/full_case_{index}/0", json=payload(case))
        elapsed = round(time.time() - start, 2)
        text = response.get_data(as_text=True)
        try:
            data = response.get_json() if response.is_json else json.loads(text)
        except Exception:
            data = {"raw": text[:500]}
        row = {
            "case": case,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "response": data,
        }
        results.append(row)
        if response.status_code == 200:
            print(f"    OK fault={data.get('fault_detected')} type={data.get('fault_type')} info={data.get('fault_info')}", flush=True)
        else:
            print(f"    ERROR {data}", flush=True)

    out_dir = Path("test_results_full_ban_goc_patch")
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "full_cases_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Full IMS and XJTU-SY Run Results",
        "",
        "| # | Group | Case | Status | Fault | Type | Stage/Info | Time (s) | Error |",
        "|---:|---|---|---:|---|---|---|---:|---|",
    ]
    for i, row in enumerate(results, start=1):
        case = row["case"]
        data = row["response"]
        md_lines.append(
            "| {i} | {group} | {label} | {status} | {fault} | {types} | {info} | {elapsed} | {error} |".format(
                i=i,
                group=case["group"],
                label=case["label"],
                status=row["status_code"],
                fault=data.get("fault_detected", ""),
                types=", ".join(data.get("fault_type", []) or []),
                info=data.get("fault_info") or data.get("analysis_result") or "",
                elapsed=row["elapsed_seconds"],
                error=str(data.get("error", ""))[:120].replace("|", "/"),
            )
        )
    md_path = out_dir / "full_cases_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    run()
