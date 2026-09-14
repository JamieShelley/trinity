from __future__ import annotations

import argparse
from pathlib import Path
import time
import tkinter as tk

from PIL import Image, ImageTk


class LiveView:
    def __init__(self, root: tk.Tk, image_path: Path, stop_path: Path) -> None:
        self.root = root
        self.image_path = image_path
        self.stop_path = stop_path
        self.last_mtime_ns = -1
        self.photo: ImageTk.PhotoImage | None = None
        self.label = tk.Label(root, text="Waiting for V16 A/B/C/F preview...")
        self.label.pack(fill="both", expand=True)
        root.title("NSAMDR V16 Live A / B / C / F")
        root.geometry("1500x650")
        self._poll()

    def _poll(self) -> None:
        if self.stop_path.exists():
            self.root.destroy()
            return
        try:
            stat = self.image_path.stat()
            if stat.st_mtime_ns != self.last_mtime_ns:
                self.last_mtime_ns = stat.st_mtime_ns
                with Image.open(self.image_path) as image:
                    view = image.convert("RGB")
                    max_w = max(self.root.winfo_width() - 20, 500)
                    max_h = max(self.root.winfo_height() - 20, 300)
                    view.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(view)
                self.label.configure(image=self.photo, text="")
        except OSError:
            pass
        self.root.after(750, self._poll)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the latest V16 A/B/C/F training contact sheet"
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    live = args.experiment_dir.resolve() / "previews" / "live"
    live.mkdir(parents=True, exist_ok=True)
    stop = live / "viewer.stop"
    stop.unlink(missing_ok=True)
    root = tk.Tk()
    LiveView(root, live / "latest_ABCF.png", stop)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
