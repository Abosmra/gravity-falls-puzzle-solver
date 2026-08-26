# Gravity Falls Puzzle Solver

An autonomous Computer Vision system for square jigsaw and tiled image puzzle reassembly using multi-cue border feature matching and the Best-Buddies placement algorithm.

The system provides an end-to-end pipeline to slice high-resolution source images into uniform grid tiles (2x2, 4x4, 8x8), apply edge-preserving enhancement, extract multi-channel boundary features, compute compatibility metrics, reconstruct the original image layout, and provide an interactive visual inspector GUI.

---

## Table of Contents
1. [Key Capabilities](#key-capabilities)
2. [Architecture and Technical Approach](#architecture-and-technical-approach)
3. [Repository Layout](#repository-layout)
4. [Installation and Setup](#installation-and-setup)
5. [Dataset Setup](#dataset-setup)
6. [CLI and Execution Guide](#cli-and-execution-guide)
7. [Visual Inspector GUI](#visual-inspector-gui)
8. [Automated Testing](#automated-testing)
9. [Output Structure](#output-structure)
10. [License and References](#license-and-references)

---

## Key Capabilities

- Unified Pipeline: Single-command end-to-end execution from raw image or dataset to reconstructed output.
- Automated Grid Tiling: Slices arbitrary images into regular grids (2x2, 4x4, 8x8) while handling odd-dimension remainder pixels gracefully.
- Edge-Preserving Preprocessing: Implements adaptive bilateral filtering, guided contour smoothing, and unsharp masking to enhance texture gradients without introducing border artifacts.
- 6-Channel Border Descriptors: Boundary feature tensors combining LAB color channels, Sobel gradient magnitude, Sobel gradient phase, and Laplacian curvature.
- Non-Linear Minkowski Distance Metric: Robust Lp (p=0.3, q=1/16) metric resistant to outliers and illumination variations.
- Best-Buddies Reassembly: Mutual best-match pairing graph formulation with constrained greedy placement, connected component segmentation, and 2-opt shift local search.
- Interactive Visual Inspector: Tkinter GUI for before/after comparison, live reloading, jump-to navigation, and asynchronous re-solving.

---

## Architecture and Technical Approach

```
+-------------------------------------------------------------+
|                        Source Image                         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     1. Tiling & Enhancement                 |
|   - Deterministic spatial grid cutting (2x2, 4x4, 8x8)      |
|   - Adaptive bilateral denoising + guided filtering         |
|   - High-frequency unsharp masking & metadata tracking      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                2. Boundary Feature Extraction               |
|   - 6-channel boundary tensors (LAB, Sobel Mag/Dir, Lap)    |
|   - 2D zero-mean unit-variance strip normalization          |
|   - Pairwise non-linear distance matrix computation         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  3. Best-Buddies Reassembly                 |
|   - Exact branch-and-bound search (2x2)                     |
|   - Mutual best-match pairing & constrained placement (NxN) |
|   - Connected component segmentation & 2-opt shift passes   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                4. Canvas Stitching & Rendering              |
|   - Seamless composite image reconstruction                 |
|   - Interactive visual inspection & live verification       |
+-------------------------------------------------------------+
```

### Feature Formulation and Distance Function

For two candidate borders A and B across corresponding orientations, the distance function is evaluated as:

D(A, B) = [ w_color * sum(|A_lab - B_lab|^p) + w_mag * sum(|A_mag - B_mag|^p) + w_dir * sum(|A_dir - B_dir|^p) + w_lap * sum(|A_lap - B_lap|^p) ]^(q/p)

Default parameters: p = 0.3, q = 1/16, w_color = 0.4, w_mag = 0.2, w_dir = 0.2, w_lap = 0.4.

---

## Repository Layout

```
gravity-falls-puzzle-solver/
|-- .gitignore                  # Git ignore definitions
|-- LICENSE                     # MIT License
|-- pyproject.toml              # Packaging and dependency specifications
|-- requirements.txt            # Python dependencies
|-- README.md                   # Project documentation
|-- main.py                     # Unified root CLI entrypoint
|-- gui.py                      # Direct visual inspector launcher
|
|-- puzzle_solver/              # Core Python package
|   |-- __init__.py             # Public exports and versioning
|   |-- __main__.py             # CLI execution module
|   |-- config.py               # Dataclasses and configuration schemas
|   |-- pipeline.py             # Multiprocessing batch pipelines and direct solvers
|   |-- core/
|   |   |-- __init__.py
|   |   |-- tiling.py           # Grid detection, enhancement, and tile slicing
|   |   |-- features.py         # Multi-channel feature extraction and distances
|   |   |-- solver.py           # Best-Buddies placement and solver logic
|   |   `-- assembly.py         # Canvas stitching and composite generation
|   `-- ui/
|       |-- __init__.py
|       `-- viewer.py           # Tkinter visual inspector GUI
|
|-- tests/                      # Automated unit and integration test suite
|   |-- __init__.py
|   |-- test_tiling.py          # Unit tests for tiling and enhancement
|   |-- test_features.py        # Unit tests for feature extraction and metrics
|   |-- test_solver.py          # Unit tests for Best-Buddies and brute-force solvers
|   |-- test_assembly.py        # Unit tests for image assembly
|   `-- test_integration.py     # End-to-end pipeline integration tests
|
|-- scripts/
|   `-- run_tests.py            # Test execution runner
|
`-- assets/                     # Demonstration assets
    |-- 2x2 Demo.png
    |-- 4x4 Demo.png
    `-- 8x8 Demo.png
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Abosmra/gravity-falls-puzzle-solver.git
   cd gravity-falls-puzzle-solver
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Setup

The benchmarking dataset is hosted on Kaggle:
[Jigsaw Puzzle Dataset on Kaggle](https://www.kaggle.com/datasets/serhiibiruk/jigsaw-puzzle)

### Directory Configuration
Extract the downloaded dataset into the `dataset_images/` folder at the root of the project:
```
dataset_images/
|-- puzzle_2x2/
|   |-- 0.jpg
|   `-- ...
|-- puzzle_4x4/
|   |-- 0.jpg
|   `-- ...
`-- puzzle_8x8/
    |-- 0.jpg
    `-- ...
```

---

## CLI and Execution Guide

### 1. Unified CLI (main.py)

#### Run Full Pipeline with Visual Inspector
```bash
python main.py all
# or simply
python main.py
```

#### Solve a Single Image End-to-End
```bash
python main.py solve --image path/to/sample.jpg --grid 4x4 --out solved_sample.png
```

#### Batch Solve Entire Dataset
```bash
python main.py solve --dataset dataset_images --out output --time-limit 60.0
```

#### Tile Extraction Only
```bash
python main.py extract --dataset dataset_images --out output/tiles
```

#### Reassembly Only
```bash
python main.py reassemble --tiles-dir output/tiles --out output/solved --group puzzle_4x4
```

#### Launch Visual Inspector GUI
```bash
python main.py gui
# or directly
python gui.py
```

---

## Visual Inspector GUI

The interactive Tkinter GUI provides real-time verification of puzzle reconstructions:

- Dual-Pane View: Displays the original/scrambled image on the left and the reconstructed solution on the right.
- Live Auto-Reload: Images update automatically in real-time as background solving processes finish.
- Direct Navigation: Jump to any specific puzzle using the grid selector and image ID input.
- Manual Re-solving: Use the 'Resolve (Redo)' button to clear previous results and run solver passes with alternative seeds.

---

## Automated Testing

The repository contains a test suite covering unit behavior and end-to-end integration:

Run all tests:
```bash
python scripts/run_tests.py
# or
python -m unittest discover tests -v
```

---

## Output Structure

- `output/tiles/<group>/<image_id>/tiles/`: Extracted PNG tiles named `tile_RR_CC.png`.
- `output/tiles/<group>/<image_id>/metadata.json`: Grid metadata containing source path, dimensions, and tile lists.
- `output/solved/<group>/<image_id>.png`: Solved and stitched composite images.

---

## License and References

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### References
- Gallagher, A. C. (2012). Jigsaw Puzzles with Pieces of Unknown Orientation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Pomeranz, D., Shemesh, M., & Ben-Shahar, O. (2011). A Fully Automated Greedy Square Jigsaw Puzzle Solver. IEEE CVPR.
