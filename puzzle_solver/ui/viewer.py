import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
import traceback
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk

from puzzle_solver.core.features import load_tiles_from_phase1
from puzzle_solver.core.solver import PuzzleSolver
from puzzle_solver.core.assembly import assemble_puzzle


class PuzzleViewerGUI:
    """Tkinter-based interactive visualizer for before/after puzzle state inspection."""

    def __init__(self, root: tk.Tk, phase1_root: str = "phase1_outputs", out_dir: str = "phase2_outputs"):
        self.root = root
        self.root.title("Gravity Falls Puzzle Solver - Visual Inspector")
        self.root.geometry("1400x900")

        self.phase1_root = phase1_root
        self.out_dir = out_dir
        self.groups_order = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]

        self.puzzles: List[Tuple[str, str]] = []
        self.current_idx: int = 0

        self.solving: bool = False
        self.current_tiles: Optional[List[Dict[str, Any]]] = None
        self.current_result: Optional[np.ndarray] = None

        self.setup_ui()
        self.scan_puzzles()

    def setup_ui(self) -> None:
        """Construct GUI frames, canvases, buttons, and status labels."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="Refresh Puzzles", command=self.scan_puzzles).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(control_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.counter_label = ttk.Label(control_frame, text="0/0")
        self.counter_label.pack(side=tk.RIGHT, padx=5)

        ttk.Button(control_frame, text="Go", command=self.goto_puzzle).pack(side=tk.RIGHT, padx=5)

        self.image_var = tk.StringVar(value="0")
        image_entry = ttk.Entry(control_frame, textvariable=self.image_var, width=6)
        image_entry.pack(side=tk.RIGHT, padx=5)
        ttk.Label(control_frame, text="Image ID:").pack(side=tk.RIGHT)

        self.grid_var = tk.StringVar(value=self.groups_order[0])
        grid_box = ttk.Combobox(
            control_frame,
            textvariable=self.grid_var,
            values=self.groups_order,
            width=12,
            state="readonly"
        )
        grid_box.pack(side=tk.RIGHT, padx=5)
        ttk.Label(control_frame, text="Grid:").pack(side=tk.RIGHT)

        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.LabelFrame(content_frame, text="Before: Original Tiles / Scrambled", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_before = tk.Canvas(left_frame, bg="gray20", width=500, height=500)
        self.canvas_before.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.LabelFrame(content_frame, text="After: Solved Reconstruction", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_after = tk.Canvas(right_frame, bg="gray20", width=500, height=500)
        self.canvas_after.pack(fill=tk.BOTH, expand=True)

        info_frame = ttk.LabelFrame(self.root, text="Puzzle Info", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_text = tk.Text(info_frame, height=3, width=100)
        self.info_text.pack(fill=tk.BOTH, expand=False)

        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)

        ttk.Button(nav_frame, text="< Previous", command=self.previous_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next >", command=self.next_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Resolve (Redo)", command=self.resolve_current).pack(side=tk.LEFT, padx=5)

    def scan_puzzles(self) -> None:
        """Scan directory structure and populate puzzle list ordered by image ID and grid size."""
        self.puzzles = []

        if not os.path.isdir(self.phase1_root):
            self.update_status(f"Phase 1 outputs not found: {self.phase1_root}")
            return

        images_by_group: Dict[str, List[str]] = {}

        for group_dir in sorted(os.listdir(self.phase1_root)):
            group_path = os.path.join(self.phase1_root, group_dir)
            if not os.path.isdir(group_path):
                continue

            images_by_group[group_dir] = []
            for item in sorted(os.listdir(group_path), key=lambda x: int(x) if x.isdigit() else x):
                item_path = os.path.join(group_path, item)
                tiles_dir = os.path.join(item_path, "tiles")
                if os.path.isdir(tiles_dir):
                    images_by_group[group_dir].append(item)

        max_image_id = 0
        for images in images_by_group.values():
            for img_id in images:
                if img_id.isdigit():
                    max_image_id = max(max_image_id, int(img_id))

        for image_id in range(max_image_id + 1):
            for group in self.groups_order:
                if group in images_by_group and str(image_id) in images_by_group[group]:
                    self.puzzles.append((group, str(image_id)))

        if self.puzzles:
            self.current_idx = 0
            self.update_status(f"Found {len(self.puzzles)} puzzles")
            self.display_puzzle()
        else:
            self.update_status("No puzzles found")

    def display_puzzle(self) -> None:
        """Render both before and after views of the active puzzle."""
        if not self.puzzles or self.current_idx >= len(self.puzzles):
            return

        group, image_id = self.puzzles[self.current_idx]
        self.update_status(f"Loading {group}/{image_id}...")
        self.root.update()

        try:
            try:
                self.current_tiles = load_tiles_from_phase1(self.phase1_root, group, image_id)
                self.show_before_image()
            except Exception:
                self.show_loading_image(self.canvas_before, "Loading tiles...")
                self.current_tiles = None

            self.show_after_image(group, image_id)
            self.update_info(group, image_id)
            self.update_counter()
            self.update_status(f"Puzzle {group}/{image_id} loaded")
        except Exception as e:
            self.update_status(f"Error: {e}")
            self.canvas_before.delete("all")
            self.canvas_after.delete("all")

    def show_before_image(self) -> None:
        """Render input source image from dataset."""
        group, image_id = self.puzzles[self.current_idx]
        dataset_folder = group if group in ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"] else "puzzle_2x2"
        original_path = os.path.join("dataset_images", dataset_folder, f"{image_id}.jpg")

        if not os.path.exists(original_path):
            original_path = os.path.join("dataset_images", dataset_folder, f"{image_id}.png")

        if os.path.exists(original_path):
            img = cv2.imread(original_path)
            if img is not None:
                self.show_image_on_canvas(self.canvas_before, img)
                return

        self.show_loading_image(self.canvas_before, "Loading original image...")

    def show_after_image(self, group: str, image_id: str) -> None:
        """Render solved puzzle image from output directory."""
        solved_path = os.path.join(self.out_dir, group, f"{image_id}.png")

        if os.path.exists(solved_path):
            img = cv2.imread(solved_path)
            if img is not None:
                self.show_image_on_canvas(self.canvas_after, img)
                self.current_result = img
                return

        self.show_loading_image(self.canvas_after, "Loading solution...")
        self.current_result = None

    def show_loading_image(self, canvas: tk.Canvas, text: str) -> None:
        """Display placeholder notification text on canvas."""
        canvas.delete("all")
        w = max(600, canvas.winfo_width())
        h = max(600, canvas.winfo_height())

        blank = np.ones((h, w, 3), dtype=np.uint8) * 200
        cv2.putText(blank, text, (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.putText(blank, "(auto-updates when ready)", (w // 2 - 170, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        self.show_image_on_canvas(canvas, blank)

    def show_image_on_canvas(self, canvas: tk.Canvas, cv_image: np.ndarray) -> None:
        """Scale and draw an OpenCV BGR image onto a Tkinter Canvas."""
        canvas_w = max(600, canvas.winfo_width())
        canvas_h = max(600, canvas.winfo_height())

        h, w = cv_image.shape[:2]
        scale = min(canvas_w / w, canvas_h / h) * 0.98
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        img_resized = cv2.resize(cv_image, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo)
        canvas.image = photo  # Keep reference

    def update_info(self, group: str, image_id: str) -> None:
        """Update the metadata text widget."""
        self.info_text.delete(1.0, tk.END)

        if "2x2" in group:
            grid_n = 2
        elif "4x4" in group:
            grid_n = 4
        elif "8x8" in group:
            grid_n = 8
        else:
            grid_n = 0

        info = f"Group: {group} | Image ID: {image_id} | Grid: {grid_n}x{grid_n}\n"

        if self.current_tiles:
            n_tiles = len(self.current_tiles)
            tile_h, tile_w = self.current_tiles[0]["img"].shape[:2]
            info += f"Tiles: {n_tiles} | Tile size: {tile_w}x{tile_h} px\n"
        else:
            info += "Tiles: Loading...\n"

        if self.current_result is not None:
            info += "Status: SOLVED\n"
        else:
            solved_path = os.path.join(self.out_dir, group, f"{image_id}.png")
            if os.path.exists(solved_path):
                info += "Status: SOLVED\n"
            else:
                info += "Status: Pending reassembly...\n"

        self.info_text.insert(tk.END, info)

    def update_counter(self) -> None:
        """Update numeric index display."""
        if self.puzzles:
            self.counter_label.config(text=f"{self.current_idx + 1}/{len(self.puzzles)}")

    def update_status(self, message: str) -> None:
        """Set status bar string."""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def solve_current(self) -> None:
        """Trigger background solver for current puzzle."""
        if self.solving:
            self.update_status("Already solving...")
            return

        if not self.current_tiles:
            self.update_status("Tiles not loaded yet...")
            return

        thread = threading.Thread(target=self._solve_worker, daemon=True)
        thread.start()

    def goto_puzzle(self) -> None:
        """Jump directly to a selected group and image ID."""
        target_group = self.grid_var.get()
        target_image = self.image_var.get().strip()
        if not target_image:
            self.update_status("Enter an image id")
            return
        if (target_group, target_image) in self.puzzles:
            self.current_idx = self.puzzles.index((target_group, target_image))
            self.display_puzzle()
            return
        self.update_status(f"Not found: {target_group}/{target_image}")

    def _solve_worker(self) -> None:
        """Background solving routine."""
        try:
            self.solving = True
            group, image_id = self.puzzles[self.current_idx]
            self.update_status(f"Solving {group}/{image_id}...")

            if "2x2" in group:
                grid_n = 2
            elif "4x4" in group:
                grid_n = 4
            elif "8x8" in group:
                grid_n = 8
            else:
                grid_n = int(np.sqrt(len(self.current_tiles)))

            solver = PuzzleSolver(self.current_tiles, grid_n, grid_n)
            result = solver.solve(time_limit=120.0)

            if result is None:
                self.update_status(f"Failed to solve {group}/{image_id}")
                return

            placement = result["placement_map"]
            out_path = Path(self.out_dir) / group / f"{image_id}.png"
            canvas = assemble_puzzle(self.current_tiles, placement, grid_n, grid_n, output_path=out_path)

            self.current_result = canvas
            score = result.get("score", 0.0)
            method = result.get("method", "unknown")

            self.update_status(f"Solved! Score: {score:.3f} ({method})")
            self.update_info(group, image_id)
            self.root.after(0, lambda: self.show_after_image(group, image_id))
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.solving = False

    def resolve_current(self) -> None:
        """Delete existing output and re-solve."""
        if not self.puzzles or self.solving:
            return
        group, image_id = self.puzzles[self.current_idx]
        solved_path = Path(self.out_dir) / group / f"{image_id}.png"
        if solved_path.exists():
            try:
                solved_path.unlink()
            except Exception as e:
                self.update_status(f"Could not delete old result: {e}")

        self.current_result = None
        self.show_loading_image(self.canvas_after, "Re-solving...")
        self.solve_current()

    def next_puzzle(self) -> None:
        """Advance to next puzzle."""
        if self.puzzles:
            self.current_idx = (self.current_idx + 1) % len(self.puzzles)
            self.display_puzzle()

    def previous_puzzle(self) -> None:
        """Step back to previous puzzle."""
        if self.puzzles:
            self.current_idx = (self.current_idx - 1) % len(self.puzzles)
            self.display_puzzle()


def launch_gui(phase1_root: str = "phase1_outputs", out_dir: str = "phase2_outputs") -> None:
    """Instantiate and execute the visual inspector Tkinter event loop."""
    root = tk.Tk()
    app = PuzzleViewerGUI(root, phase1_root=phase1_root, out_dir=out_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
