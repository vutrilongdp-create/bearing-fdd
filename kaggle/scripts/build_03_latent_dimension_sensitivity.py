"""Build the standalone Kaggle notebook for bottleneck-dimension sensitivity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "kaggle_bearing_fdd.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "03_latent_dimension_sensitivity.ipynb"


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
        """# BEARING-FDD: Full-Sample Latent-Dimension Sensitivity

Notebook này kiểm tra giả thuyết rằng bottleneck một chiều làm mất quá nhiều
thông tin của waveform IMS-2.

```text
IS-3500-700-200-k-200-700-3500-IS
```

- Thử `k = 1, 8, 16, 32, 64`.
- Decoder tái tạo từ toàn bộ vector `z` có `k` chiều.
- HI của mỗi mẫu là reconstruction MSE toàn mẫu, không phải một tọa độ latent.
- P95 được tính trên 300 HI khỏe mạnh.
- FFP là mẫu đầu tiên của năm HI liên tiếp vượt P95.

Đây là thí nghiệm mới của project, không phải tái lập chính xác Table 1 của bài
báo. Nó cũng khác baseline cửa sổ 2.048 điểm, nơi HI là P95 của nhiều window MSE.
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
    markdown("## Pipeline được nhúng trực tiếp"),
    code(module_source),
    markdown(
        """## 1. Gắn dữ liệu

Trên Kaggle, chọn **Add Input** và gắn dataset chứa `IMS2.csv`.
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
    markdown("## 2. Cấu hình thí nghiệm"),
    code(
        """LATENT_DIMENSIONS = (1, 8, 16, 32, 64)
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
    model_variant="multilatent_full_sample",
    latent_dim=1,
    lambda_monotonic=0.0,
    lambda_smoothing=0.0,
    force_retrain=True,
    run_xai=False,
    xai_scope="all",
    run_diagnosis=False,
)

print(json.dumps(asdict(base_config), indent=2))
print("Số trường hợp:", len(LATENT_DIMENSIONS) * len(SEEDS))
"""
    ),
    markdown(
        """## 3. Preflight dữ liệu và bộ nhớ

Phần lớn tham số nằm ở hai lớp nối với input 20.480 chiều, nên tăng `k` chỉ làm
tăng tương đối ít tham số. Nếu OOM, giảm batch size; không đổi kiến trúc âm thầm.
"""
    ),
    code(
        """paths = discover_data(base_config)
report = validate_dataset(paths["IMS2"]["data"], DATASETS["IMS2"])
assert report["rows"] == 984
assert report["dimension"] == 20480

preflight_rows = []
for latent_dim in LATENT_DIMENSIONS:
    parameters = estimate_ms2ae_parameters(report["dimension"], latent_dim)
    preflight_rows.append(
        {
            "latent_dim": latent_dim,
            "rows": report["rows"],
            "dimension": report["dimension"],
            "model_parameters": parameters,
            "weights_fp32_gb": parameters * 4 / 1e9,
            "rough_training_state_gb": parameters * 4 * 4 / 1e9,
        }
    )

display(pd.DataFrame(preflight_rows))
print("Preflight passed.")
"""
    ),
    markdown(
        """## 4. Chạy latent-dimension sensitivity

`resume=True` bỏ qua case đã hoàn thành. Model nặng được xóa sau đánh giá, nhưng
config, training history, HI CSV/plot và result JSON vẫn được giữ.
"""
    ),
    code(
        """SENSITIVITY_ROOT = "/kaggle/working/bearing_fdd_latent_sensitivity"

summary = run_latent_dimension_sensitivity(
    base_config,
    LATENT_DIMENSIONS,
    seeds=SEEDS,
    sensitivity_root=SENSITIVITY_ROOT,
    resume=True,
    keep_models=False,
)

display(summary)
"""
    ),
    markdown("## 5. Kiểm tra và xếp hạng"),
    code(
        """completed = summary[summary["status"] == "completed"].copy()
columns = [
    "case_id",
    "latent_dim",
    "seed",
    "threshold_p95",
    "project_ffp_zero_based",
    "project_ffp_one_based",
    "paper_ffp_one_based",
    "difference_one_based",
    "absolute_difference_one_based",
    "final_training_reconstruction_loss",
    "hi_standard_deviation",
    "decreasing_step_ratio",
    "second_difference_rmse",
    "elapsed_seconds",
]
display(
    completed.sort_values(
        "absolute_difference_one_based",
        na_position="last",
    )[columns]
)

assert len(completed) == len(LATENT_DIMENSIONS) * len(SEEDS)
assert completed["hi_definition"].str.contains(
    "full-sample reconstruction MSE"
).all()
print("All latent-dimension cases completed.")
"""
    ),
    markdown("## 6. Xem đường HI"),
    code(
        """from IPython.display import Image, display

for case_root in sorted(Path(SENSITIVITY_ROOT).glob("case_*")):
    figure = case_root / "figures" / "IMS2_hi.png"
    if figure.exists():
        print(case_root.name)
        display(Image(filename=str(figure), width=650))
"""
    ),
    markdown("## 7. Nén kết quả"),
    code(
        """import shutil

archive = shutil.make_archive(
    "/kaggle/working/bearing_fdd_latent_sensitivity",
    "zip",
    root_dir=SENSITIVITY_ROOT,
)
print("Download from Kaggle Output:", archive)
"""
    ),
    markdown(
        """## 8. Quy tắc chọn bước tiếp theo

Không chọn `k` chỉ vì một lần chạy có FFP gần #536. Trước hết kiểm tra:

- FFP tồn tại và không phải một spike ngẫu nhiên;
- HI sau FFP duy trì vượt threshold hợp lý;
- reconstruction loss không tăng bất thường;
- HI không co cụm thành gần hằng số;
- kết quả ổn định khi chạy lại seed `(7, 42, 123)`.

Chỉ bật XAI và chẩn đoán sau khi một hoặc hai kích thước vượt qua kiểm tra nhiều
seed. Kết quả của biến thể này phải được báo riêng với windowed baseline và
regularized scalar-latent variant.
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
        "language_info": {"name": "Python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(NOTEBOOK_PATH.name)
