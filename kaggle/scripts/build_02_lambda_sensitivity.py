"""Build the standalone Kaggle notebook for IMS-2 lambda sensitivity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "kaggle_bearing_fdd.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "02_lambda_sensitivity.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


module_source = MODULE_PATH.read_text(encoding="utf-8")
module_source = module_source.rsplit('\nif __name__ == "__main__":', 1)[0] + "\n"

cells = [
    markdown(
        """# BEARING-FDD: IMS-2 Lambda Sensitivity

Notebook này sàng lọc độc lập ảnh hưởng của `lambda_monotonic` và
`lambda_smoothing` đối với HI/P95/FFP của IMS-2.

- Mười trường hợp dùng cùng dữ liệu, seed, kiến trúc và cấu hình Table 1.
- Mỗi trường hợp có thư mục riêng và có thể tiếp tục sau khi session bị ngắt.
- Giai đoạn sàng lọc tắt XAI và chẩn đoán để giảm thời gian.
- Model nặng được xóa sau đánh giá; HI, log, hình và JSON vẫn được giữ.
- `project_ffp_one_based` mới là giá trị so sánh trực tiếp với FFP #536 của bài báo.

Các lambda và công thức regularization là lựa chọn của project; bài báo không
công bố chúng.
"""
    ),
    code(
        """import os
import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Kaggle:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local/not detected"))
"""
    ),
    markdown(
        """## Pipeline

Cell dưới đây nhúng toàn bộ source để notebook chạy độc lập khi upload lên Kaggle.
"""
    ),
    code(module_source),
    markdown(
        """## 1. Gắn dữ liệu

Trên Kaggle, chọn **Add Input** và gắn dataset chứa `IMS2.csv`. Không dán URL
notebook hoặc URL từng file vào `INPUT_ROOT`.
"""
    ),
    code(
        """ATTACHED_INPUT_ROOT = "/kaggle/input"
KAGGLE_DATASET_URL = ""


def resolve_input_root(attached_root, dataset_url=""):
    dataset_url = dataset_url.strip().rstrip("/")
    if not dataset_url:
        root = Path(attached_root)
        if not root.exists():
            raise FileNotFoundError(f"Không tồn tại thư mục dữ liệu: {root}")
        return str(root)

    marker = "kaggle.com/datasets/"
    if marker not in dataset_url:
        raise ValueError(
            "URL phải có dạng https://www.kaggle.com/datasets/<owner>/<slug>"
        )
    handle = dataset_url.split(marker, 1)[1].split("?")[0]
    parts = [part for part in handle.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"Không đọc được owner/slug từ URL: {dataset_url}")
    import kagglehub

    return str(kagglehub.dataset_download("/".join(parts)))


INPUT_ROOT = resolve_input_root(ATTACHED_INPUT_ROOT, KAGGLE_DATASET_URL)
print("INPUT_ROOT:", INPUT_ROOT)
for path in sorted(Path(INPUT_ROOT).rglob("*IMS2*.csv")):
    print(path, f"{path.stat().st_size / 1e9:.2f} GB")
"""
    ),
    markdown(
        """## 2. Ma trận thử nghiệm

Ba mức `1e-6`, `1e-4`, `1e-2` tạo phép sàng lọc theo thang log:

1. Không regularization.
2. Chỉ monotonic.
3. Chỉ smoothing.
4. Hai regularizer cùng mức.

Trường hợp `(0.01, 0.01)` đã tạo FFP rất muộn được giữ lại để đối chứng.
"""
    ),
    code(
        """LAMBDA_CASES = (
    (0.0, 0.0),
    (1e-6, 0.0),
    (1e-4, 0.0),
    (1e-2, 0.0),
    (0.0, 1e-6),
    (0.0, 1e-4),
    (0.0, 1e-2),
    (1e-6, 1e-6),
    (1e-4, 1e-4),
    (1e-2, 1e-2),
)

# Sàng lọc bằng một seed. Sau khi chọn 2-3 cấu hình tốt, phải chạy lại nhiều seed.
SEEDS = (42,)

base_config = RunConfig(
    input_root=INPUT_ROOT,
    output_root="/kaggle/working/unused_base_output",
    datasets=("IMS2",),
    healthy_samples=300,
    epochs=5,
    batch_size=64,
    eval_chunk=40,
    consecutive=5,
    random_seed=42,
    model_variant="regularized_full_sample",
    lambda_monotonic=0.0,
    lambda_smoothing=0.0,
    force_retrain=True,
    run_xai=False,
    xai_scope="all",
    run_diagnosis=False,
)

print(json.dumps(asdict(base_config), indent=2))
print("Số trường hợp:", len(LAMBDA_CASES) * len(SEEDS))
"""
    ),
    markdown("## 3. Preflight"),
    code(
        """paths = discover_data(base_config)
report = validate_dataset(paths["IMS2"]["data"], DATASETS["IMS2"])
parameters = estimate_ms2ae_parameters(report["dimension"])
report["model_parameters"] = parameters
report["weights_fp32_gb"] = parameters * 4 / 1e9
report["rough_training_state_gb"] = parameters * 4 * 4 / 1e9
display(pd.DataFrame([report]))

assert report["dimension"] == 20480
assert report["rows"] >= 984
assert base_config.batch_size >= 3
print("Preflight passed.")
"""
    ),
    markdown(
        """## 4. Chạy sensitivity

`resume=True` sẽ bỏ qua các case đã hoàn thành nếu session bị ngắt. Không đổi
thứ tự hoặc nội dung `LAMBDA_CASES` khi đang resume cùng một output.
"""
    ),
    code(
        """SENSITIVITY_ROOT = "/kaggle/working/bearing_fdd_lambda_sensitivity"

summary = run_lambda_sensitivity(
    base_config,
    LAMBDA_CASES,
    seeds=SEEDS,
    sensitivity_root=SENSITIVITY_ROOT,
    resume=True,
    keep_models=False,
)

display(summary)
"""
    ),
    markdown("## 5. Xếp hạng và kiểm tra"),
    code(
        """completed = summary[summary["status"] == "completed"].copy()
ranking_columns = [
    "case_id",
    "lambda_monotonic",
    "lambda_smoothing",
    "seed",
    "threshold_p95",
    "project_ffp_zero_based",
    "project_ffp_one_based",
    "paper_ffp_one_based",
    "difference_one_based",
    "absolute_difference_one_based",
    "weighted_monotonic_to_reconstruction",
    "weighted_smoothing_to_reconstruction",
    "hi_standard_deviation",
    "decreasing_step_ratio",
    "second_difference_rmse",
]
ranking = completed.sort_values(
    [
        "absolute_difference_one_based",
        "decreasing_step_ratio",
        "second_difference_rmse",
    ],
    na_position="last",
)
display(ranking[ranking_columns])

assert len(completed) == len(LAMBDA_CASES) * len(SEEDS), (
    "Có case chưa hoàn thành; xem sensitivity_case.json trong thư mục case."
)
print("All sensitivity cases completed.")
"""
    ),
    markdown(
        """## 6. Nén kết quả

File ZIP chứa config, loss, HI CSV/plot, audit và bảng tổng hợp của từng case.
"""
    ),
    code(
        """import shutil

archive = shutil.make_archive(
    "/kaggle/working/bearing_fdd_lambda_sensitivity",
    "zip",
    root_dir=SENSITIVITY_ROOT,
)
print("Download from Kaggle Output:", archive)
"""
    ),
    markdown(
        """## 7. Xác nhận bằng nhiều seed

Không chọn lambda chỉ vì một FFP gần #536. Sau vòng sàng lọc, giữ 2-3 cấu hình
tốt và chạy lại với `SEEDS = (7, 42, 123)`. Đánh giá đồng thời:

- sai lệch FFP;
- đóng góp loss có trọng số so với reconstruction loss;
- độ phân tán HI và nguy cơ HI co cụm quanh 0.5;
- tỷ lệ bước giảm;
- RMSE sai phân bậc hai.

Chỉ bật XAI và chẩn đoán cho cấu hình đã vượt qua bước xác nhận nhiều seed.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(NOTEBOOK_PATH.name)
