"""Build the self-contained Kaggle notebook from the maintained Python module."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "kaggle_bearing_fdd.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "01_bearing_fdd_pipeline.ipynb"


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
module_source = module_source.rsplit("\nif __name__ == \"__main__\":", 1)[0] + "\n"

cells = [
    markdown(
        """# BEARING-FDD: Full Kaggle Reproduction

This notebook retrains the project's dense MS2AE implementation and reruns the
full evaluation pipeline:

1. Validate IMS and XJTU-SY CSV files.
2. Train `IS-3500-700-200-1-200-700-3500-IS` for five epochs.
3. Compute sample HI as P95 of window-level reconstruction MSE.
4. Compute the healthy P95 threshold and FFP using five consecutive samples.
5. Compute engineering-feature Pearson correlations.
6. Isolate representative faulty samples, select a Kurtogram band, run
   bandpass + envelope FFT, and match BPFO/BPFI/BSF/FTF harmonics.

Start with IMS-2 only. Run all seven datasets after that test succeeds.
"""
    ),
    code(
        """# Runtime information
import os
import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Kaggle:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local/not detected"))
"""
    ),
    markdown(
        """## Pipeline implementation

The cell below is intentionally self-contained, so this `.ipynb` can be
uploaded without a separate source package. The maintained copy is
`kaggle_bearing_fdd.py`.
"""
    ),
    code(module_source),
    markdown(
        """## 1. Gắn dữ liệu đầu vào

### Cách khuyến nghị: dùng **Add Input**

1. Trên Kaggle, mở notebook này.
2. Ở bảng bên phải chọn **Add Input**.
3. Tìm và chọn Kaggle Dataset chứa các file CSV.
4. Giữ nguyên `ATTACHED_INPUT_ROOT = "/kaggle/input"`.

Kaggle tự gắn dữ liệu vào:

```text
/kaggle/input/<dataset-slug>/
```

Notebook tìm file CSV theo kiểu đệ quy, vì vậy **không cần** nhập tên
`<dataset-slug>` và **không dán URL web vào `input_root`**.

### Cách tùy chọn: dùng URL Kaggle Dataset

Chỉ điền URL dạng:

```text
https://www.kaggle.com/datasets/<owner>/<dataset-slug>
```

Phương án này dùng `kagglehub.dataset_download()` và có thể tải lặp lại vài GB
dữ liệu. Chỉ dùng khi không thể chọn **Add Input**. Dataset riêng tư có thể yêu
cầu quyền truy cập hoặc Kaggle API token.
"""
    ),
    code(
        """# ====================== CHỈNH DỮ LIỆU Ở ĐÂY ======================
# CÁCH 1 - KHUYẾN NGHỊ:
# Sau khi chọn "Add Input" trên Kaggle, giữ nguyên hai dòng dưới đây.
ATTACHED_INPUT_ROOT = "/kaggle/input"
KAGGLE_DATASET_URL = ""

# CÁCH 2 - TÙY CHỌN:
# Nếu không dùng Add Input, dán URL dataset vào KAGGLE_DATASET_URL.
# Ví dụ:
# KAGGLE_DATASET_URL = (
#     "https://www.kaggle.com/datasets/ten-tai-khoan/bearing-fdd-data"
# )
#
# Không điền link Google Drive, link notebook hoặc link từng file CSV.

def resolve_input_root(attached_root, dataset_url=""):
    \"\"\"Trả về thư mục chứa dữ liệu, không trả về URL web.\"\"\"
    dataset_url = dataset_url.strip().rstrip("/")
    if not dataset_url:
        root = Path(attached_root)
        if not root.exists():
            raise FileNotFoundError(
                f"Không tồn tại {root}. Trên Kaggle hãy chọn Add Input; "
                "khi chạy local hãy thay ATTACHED_INPUT_ROOT bằng đường dẫn data."
            )
        print(f"[INPUT] Dùng dữ liệu đã gắn: {root}")
        return str(root)

    marker = "kaggle.com/datasets/"
    if marker not in dataset_url:
        raise ValueError(
            "KAGGLE_DATASET_URL phải có dạng "
            "https://www.kaggle.com/datasets/<owner>/<slug>"
        )
    handle = dataset_url.split(marker, 1)[1].split("?")[0]
    parts = [part for part in handle.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"Không đọc được owner/slug từ URL: {dataset_url}")

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Thiếu kagglehub. Chạy `%pip install kagglehub` hoặc dùng Add Input."
        ) from exc

    print(f"[INPUT] Đang tải Kaggle Dataset: {'/'.join(parts)}")
    downloaded = kagglehub.dataset_download("/".join(parts))
    print(f"[INPUT] Dữ liệu tải về: {downloaded}")
    return str(downloaded)


INPUT_ROOT = resolve_input_root(ATTACHED_INPUT_ROOT, KAGGLE_DATASET_URL)

# Hiển thị nhanh các CSV mà notebook nhìn thấy.
visible_csv = sorted(Path(INPUT_ROOT).rglob("*.csv"))
print(f"[INPUT] Tìm thấy {len(visible_csv)} file CSV")
for path in visible_csv[:30]:
    print(" -", path.relative_to(Path(INPUT_ROOT)), f"({path.stat().st_size / 1e9:.2f} GB)")
if len(visible_csv) > 30:
    print(f" ... và {len(visible_csv) - 30} file khác")
"""
    ),
    markdown(
        """## 2. Cấu hình thí nghiệm

- `model_variant="regularized_full_sample"`: dùng toàn bộ mẫu làm đầu vào và
  áp dụng loss đơn điệu/làm mượt thật.
- `model_variant="dense_windowed_baseline"`: giữ baseline cửa sổ 2.048 điểm,
  không có ràng buộc chuỗi.
- `force_retrain=True`: huấn luyện model mới từ đầu.
- `xai_scope="all"`: tái hiện cách tính bảng XAI hiện tại của project.
- `xai_scope="faulty"`: chỉ tính XAI từ FFP trở đi.
- Notebook sẽ dùng `fastdtw` nếu môi trường đã cài thư viện. Nếu chưa có,
  kết quả ghi rõ `direct_difference_fastdtw_unavailable`; không được gọi kết
  quả này là DTW.

Để dùng DTW đúng trong bước chẩn đoán, bật Internet của Kaggle và chạy một cell:

```python
%pip install fastdtw==0.3.4
```

Sau khi cài, chọn **Restart Session** rồi chạy lại toàn bộ notebook.

### Hàm mất mát của biến thể regularized

Với các mẫu trong batch được giữ đúng thứ tự thời gian:

```text
L = L_reconstruction
  + lambda_monotonic * [L_mon(features_3500) + L_mon(HI)]
  + lambda_smoothing * [L_smooth(features_200) + L_smooth(HI)]
```

```text
L_mon(z)    = mean(ReLU(z_t - z_(t+1)))
L_smooth(z) = mean((z_(t+1) - 2*z_t + z_(t-1))^2)
```

`L_mon` phạt bước giảm khi HI được quy ước tăng theo suy giảm. `L_smooth`
phạt sai phân bậc hai, giúp đường cong ít dao động nhưng vẫn cho phép xu hướng
tuyến tính. Bài báo không công bố các công thức hoặc hệ số lambda này; đây là
hiện thực toán học minh bạch của project, không phải khẳng định code chính xác
tuyệt đối của tác giả.
"""
    ),
    code(
        """# ====================== CHỈNH THÍ NGHIỆM Ở ĐÂY ======================
# Chạy IMS-2 trước để kiểm tra toàn bộ pipeline.
DATASETS_TO_RUN = ("IMS2",)

# Sau khi IMS-2 chạy thành công, dùng dòng sau để chạy đủ bảy tập:
# DATASETS_TO_RUN = tuple(DATASETS)

config = RunConfig(
    # Đây phải là đường dẫn thư mục, KHÔNG phải URL Kaggle.
    input_root=INPUT_ROOT,

    # Mọi model, log, bảng và hình sẽ được lưu trong thư mục này.
    output_root="/kaggle/working/bearing_fdd_results",

    datasets=DATASETS_TO_RUN,

    # 300 mẫu đầu được giả định là khỏe mạnh và dùng để train/tính P95.
    healthy_samples=300,

    # Lựa chọn kỹ thuật của project: chia mỗi mẫu thành cửa sổ 2.048 điểm.
    window_size=2048,

    # Tham số Table 1: Adam lr=0.001, 5 epochs, batch size 64.
    epochs=5,
    batch_size=64,

    # Số mẫu đánh giá mỗi lượt. Giảm xuống 20 nếu Kaggle thiếu RAM.
    eval_chunk=40,

    # FFP là mẫu đầu tiên trong 5 giá trị HI liên tiếp vượt ngưỡng P95.
    consecutive=5,

    # Seed giúp lần chạy có khả năng tái lập tốt hơn.
    random_seed=42,

    # "regularized_full_sample": IS là toàn bộ 20.480/32.768 điểm; có loss
    # monotonic và smoothing; HI là nút sigmoid một chiều.
    # "dense_windowed_baseline": nhẹ hơn nhưng không có ràng buộc toán học.
    model_variant="regularized_full_sample",

    # Bài báo không công bố hai hệ số này. Phải lưu chúng trong run_config.json
    # và nên làm sensitivity analysis trước khi đưa ra kết luận.
    lambda_monotonic=0.01,
    lambda_smoothing=0.01,

    # True: luôn train model mới. False: dùng model đã có trong Output nếu có.
    force_retrain=True,

    # XAI dùng tương quan Pearson với các đặc trưng kỹ thuật.
    run_xai=True,

    # "all": giống bảng kết quả project hiện tại.
    # "faulty": chỉ tính từ FFP trở đi.
    xai_scope="all",

    # Chạy DTW/baseline difference -> Kurtogram -> Envelope FFT -> harmonic.
    run_diagnosis=True,

    # Một peak được xem là khớp nếu nằm trong ±8 Hz quanh harmonic lý thuyết.
    harmonic_tolerance_hz=8.0,

    # Kiểm tra từ 1X đến 6X cho BPFO/BPFI/BSF/FTF.
    max_harmonic=6,
)

print(json.dumps(asdict(config), indent=2))
"""
    ),
    markdown(
        """## 3. Smoke test cho công thức regularization

Cell này kiểm tra trực tiếp công thức và gradient trên dữ liệu tổng hợp nhỏ.
Không tiếp tục train dữ liệu thật nếu bất kỳ `assert` nào thất bại.
"""
    ),
    code(
        """# Chuỗi tăng phải không bị phạt monotonicity.
increasing = tf.constant([[0.0], [1.0], [2.0], [3.0]])
assert float(monotonic_increasing_loss(increasing).numpy()) == 0.0

# Chuỗi có bước giảm phải nhận loss dương.
with_decrease = tf.constant([[0.0], [1.0], [0.5], [2.0]])
assert float(monotonic_increasing_loss(with_decrease).numpy()) > 0.0

# Chuỗi tuyến tính có sai phân bậc hai bằng 0.
assert float(second_difference_smoothing_loss(increasing).numpy()) == 0.0

# Kiểm tra toàn bộ loss tạo được gradient. input_dim=16 chỉ dùng cho smoke test.
tiny_backbone, tiny_autoencoder, tiny_encoder = build_regularized_ms2ae(16)
tiny_batch = tf.random.normal((4, 16), seed=config.random_seed)
with tf.GradientTape() as tape:
    tiny_outputs = tiny_backbone(tiny_batch, training=True)
    tiny_losses = regularized_loss_components(
        tiny_batch,
        *tiny_outputs,
        lambda_monotonic=config.lambda_monotonic,
        lambda_smoothing=config.lambda_smoothing,
    )
tiny_gradients = tape.gradient(
    tiny_losses["loss"], tiny_backbone.trainable_variables
)
assert np.isfinite(float(tiny_losses["loss"].numpy()))
assert all(gradient is not None for gradient in tiny_gradients)

print("Regularization smoke test passed.")
del tiny_backbone, tiny_autoencoder, tiny_encoder, tiny_gradients
cleanup_memory()
"""
    ),
    markdown(
        """## 4. Kiểm tra dữ liệu trước khi train

Cell này phải chạy thành công trước khi huấn luyện. Kích thước mong đợi:

- IMS: 20.480 điểm/mẫu.
- XJTU-SY: 32.768 điểm/mẫu = 25.600 Hz × 1,28 s.

Nếu báo thiếu file, kiểm tra lại **Add Input**, tên file và tránh gắn nhiều
dataset có các file trùng tên.
"""
    ),
    code(
        """data_paths = discover_data(config)
validation_rows = []
for dataset in config.datasets:
    report = validate_dataset(data_paths[dataset]["data"], DATASETS[dataset])
    if config.model_variant == "regularized_full_sample":
        parameters = estimate_ms2ae_parameters(report["dimension"])
        report["model_parameters"] = parameters
        report["weights_fp32_gb"] = parameters * 4 / 1e9
        # Adam slots, gradients and weights require several times this number.
        report["rough_training_state_gb"] = parameters * 4 * 4 / 1e9
    validation_rows.append({"dataset": dataset, **report})

validation = pd.DataFrame(validation_rows)
display(validation)

assert set(validation["dataset"]) == set(config.datasets)
assert (validation["rows"] >= config.healthy_samples + config.consecutive).all()
print("Preflight checks passed.")
if config.model_variant == "regularized_full_sample":
    print(
        "Lưu ý: full-sample Dense MS2AE rất lớn. Nếu GPU hết bộ nhớ, "
        "giảm batch_size trước; không âm thầm chuyển sang baseline."
    )
"""
    ),
    markdown(
        """## 5. Huấn luyện và đánh giá

Mỗi dataset được xử lý lần lượt. Sau khi lưu model và kết quả, TensorFlow
session được xóa để giải phóng RAM/GPU trước dataset tiếp theo.
"""
    ),
    code(
        """results = run_pipeline(config)
summary = pd.read_csv(Path(config.output_root) / "summary.csv")
display(summary)
"""
    ),
    markdown("## 6. Kiểm tra tính toàn vẹn của kết quả"),
    code(
        """for result in results:
    assert np.isfinite(result["threshold_p95"])
    assert result["threshold_p95"] >= 0
    assert result["hi_definition"] in {
        "scalar sigmoid bottleneck with sequence regularization",
        "P95 of window reconstruction MSE",
    }
    if result["ffp"] is not None:
        assert result["ffp"] >= config.healthy_samples
    print(
        result["dataset"],
        "Variant=", result["model_variant"],
        "P95=", result["threshold_p95"],
        "FFP=", result["ffp"],
        "Paper FFP=", result["paper_ffp"],
        "HI decreasing ratio=",
        result["monotonicity_audit"]["decreasing_step_ratio"],
    )

print("Integrity checks passed.")
"""
    ),
    markdown("## 7. Xem nhanh các hình đã tạo"),
    code(
        """from IPython.display import Image, display

figure_dir = Path(config.output_root) / "figures"
for figure_path in sorted(figure_dir.glob("*.png")):
    print(figure_path.name)
    display(Image(filename=str(figure_path), width=650))
"""
    ),
    markdown("## 8. Nén toàn bộ kết quả để tải xuống"),
    code(
        """import shutil

archive = shutil.make_archive(
    "/kaggle/working/bearing_fdd_results",
    "zip",
    root_dir=config.output_root,
)
print("Download from Kaggle Output:", archive)
"""
    ),
    markdown(
        """## Full seven-dataset run

After IMS-2 succeeds, replace `DATASETS_TO_RUN` with:

```python
DATASETS_TO_RUN = tuple(DATASETS)
```

Then restart the Kaggle session and run all cells. Restarting prevents the
IMS-2 test model from occupying GPU memory before the full run.
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
