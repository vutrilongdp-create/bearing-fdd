"""Static validation for the self-contained Kaggle package."""

from __future__ import annotations

import ast
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
module_path = root / "src" / "kaggle_bearing_fdd.py"
notebook_paths = (
    root / "notebooks" / "01_bearing_fdd_pipeline.ipynb",
    root / "notebooks" / "02_lambda_sensitivity.ipynb",
    root / "notebooks" / "03_latent_dimension_sensitivity.ipynb",
)

module_source = module_path.read_text(encoding="utf-8")
ast.parse(module_source, filename=str(module_path))
expected_inline = module_source.rsplit('\nif __name__ == "__main__":', 1)[0] + "\n"

for notebook_path in notebook_paths:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    for index, cell in enumerate(code_cells):
        compile(
            "".join(cell["source"]),
            f"{notebook_path.name}-cell-{index}",
            "exec",
        )

    actual_inline = "".join(notebook["cells"][3]["source"])
    if actual_inline != expected_inline:
        raise RuntimeError(
            f"{notebook_path.name} pipeline cell is stale; rebuild the notebook"
        )

    print(
        f"OK: {notebook_path.name}: {len(notebook['cells'])} cells, "
        f"{len(code_cells)} code cells"
    )

print(f"OK: {len(module_source.splitlines())} module lines")
