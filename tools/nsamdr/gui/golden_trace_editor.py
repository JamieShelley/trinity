"""G0 independent contour tracing inside the existing NSAMDR operator GUI.

Operator clicks form native-coordinate control points. The tool never derives
a clean reference by detecting edges or sharpening the source. Every stroke
has human ownership and an exact frozen source hash.
"""
from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from tools.nsamdr.neural.v178.golden_patch import (
    TRACE_SCHEMA, render_contour_trace,
)


def open_trace_editor(parent: tk.Misc, folder: Path) -> None:
    from PIL import Image, ImageTk

    folder = Path(folder).resolve()
    evidence = json.loads((folder/"evidence.json").read_text(encoding="utf-8"))
    if not (folder/"native_albedo.png").is_file():
        raise FileNotFoundError("golden-patch native albedo not prepared")
    size = int(evidence["identity"]["nativeRoi"][2])
    zoom = 4
    window = tk.Toplevel(parent)
    window.title("NSAMDR — G0 independent clean-contour trace")
    window.geometry("1250x900")
    window.minsize(850, 600)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    author = tk.StringVar()
    width = tk.StringVar(value="0.75")
    current: list[list[float]] = []
    entries: list[dict] = []
    current_path = folder/"trace.json"
    if current_path.exists():
        existing = json.loads(current_path.read_text(encoding="utf-8"))
        if existing.get("sourceSha256") != evidence["identity"]["sourceSha256"]:
            raise RuntimeError("previous trace does not match this frozen source")
        entries.extend(existing.get("paths", []))
        author.set(existing.get("author", ""))

    toolbar = ttk.Frame(window, padding=8)
    toolbar.grid(row=0, column=0, sticky="ew")
    ttk.Label(toolbar, text="Trace author:").pack(side="left")
    ttk.Entry(toolbar, textvariable=author, width=20).pack(side="left", padx=4)
    ttk.Label(toolbar, text="Width (native px):").pack(side="left", padx=(10,0))
    ttk.Entry(toolbar, textvariable=width, width=7).pack(side="left", padx=4)

    frame = ttk.Frame(window)
    frame.grid(row=1, column=0, sticky="nsew")
    frame.columnconfigure(0,weight=1)
    frame.rowconfigure(0,weight=1)
    canvas = tk.Canvas(frame, background="#222222", highlightthickness=0)
    scroll_y = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=scroll_y.set,xscrollcommand=scroll_x.set)
    canvas.grid(row=0,column=0,sticky="nsew")
    scroll_y.grid(row=0,column=1,sticky="ns")
    scroll_x.grid(row=1,column=0,sticky="ew")
    source_image = Image.open(folder/"native_albedo.png").convert("RGB")
    displayed=source_image.resize((size*zoom,size*zoom),Image.Resampling.NEAREST)
    photo=ImageTk.PhotoImage(displayed,master=window)
    canvas.create_image(0,0,image=photo,anchor="nw",tags=("background",))
    canvas.image_reference=photo
    canvas.configure(scrollregion=(0,0,size*zoom,size*zoom))

    status=tk.StringVar(value="Click points along one clean line centre; finish a line to start another.")
    ttk.Label(window,textvariable=status,padding=8).grid(row=2,column=0,sticky="ew")

    def draw() -> None:
        canvas.delete("stroke")
        for shape in entries:
            pts=shape["points"]
            xy=[c*zoom for p in pts for c in p]
            if len(xy)>=4:
                canvas.create_line(*xy,fill="#6df3aa",width=max(1.,float(shape["widthNative"])*zoom),
                                   smooth=False,tags=("stroke",))
        xy=[c*zoom for p in current for c in p]
        if len(xy)>=4:
            canvas.create_line(*xy,fill="#fff176",width=max(1.,float(width.get() or "1")*zoom),
                               smooth=False,tags=("stroke",))
        for x,y in current:
            px,py=x*zoom,y*zoom
            canvas.create_oval(px-3,py-3,px+3,py+3,outline="#ffec83",tags=("stroke",))
        status.set(f"Completed contours: {len(entries)} | points in current line: {len(current)}")

    def clicked(event) -> None:
        x=canvas.canvasx(event.x)/zoom
        y=canvas.canvasy(event.y)/zoom
        if 0<=x<size and 0<=y<size:
            current.append([round(x,3),round(y,3)])
            draw()

    def finish() -> None:
        if len(current)<2:
            messagebox.showinfo("Golden contour", "Select at least two points.",parent=window)
            return
        try:
            value=float(width.get())
            if not 0.1<=value<=8:
                raise ValueError("width must be 0.1..8 native pixels")
        except ValueError as exc:
            messagebox.showerror("Contour width",str(exc),parent=window)
            return
        entries.append({"points":[p[:] for p in current],"widthNative":value})
        current.clear()
        draw()

    def undo() -> None:
        if current:
            current.pop()
        elif entries:
            previous=entries.pop()
            current.extend(previous["points"])
            width.set(str(previous["widthNative"]))
        draw()

    def save() -> None:
        if current:
            messagebox.showinfo("Golden contour","Finish the active line before saving.",parent=window)
            return
        if not entries or not author.get().strip():
            messagebox.showerror("Golden contour",
                                 "An independent author and at least one traced line are required.",
                                 parent=window)
            return
        payload={
            "schema":TRACE_SCHEMA,
            "sourceSha256":evidence["identity"]["sourceSha256"],
            "author":author.get().strip(),
            "method":"independent-human-contour-trace",
            "paths":entries,
            "notes":"Manually traced from native Raven crop; not derived by an edge detector.",
        }
        # No output is claimed until full provenance checks pass.
        previous=current_path.read_bytes() if current_path.exists() else None
        current_path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
        try:
            result=render_contour_trace(folder)
        except Exception as exc:
            if previous is not None:
                current_path.write_bytes(previous)
            else:
                current_path.unlink(missing_ok=True)
            messagebox.showerror("Trace validation failed",str(exc),parent=window)
            return
        messagebox.showinfo("Independent contour reference",
                            f"Saved:\n{result}\n\nReference not yet qualified for production.",
                            parent=window)
    ttk.Button(toolbar,text="Finish line",command=finish).pack(side="left",padx=5)
    ttk.Button(toolbar,text="Undo point/line",command=undo).pack(side="left",padx=5)
    ttk.Button(toolbar,text="Save curated reference",command=save).pack(side="left",padx=5)
    canvas.bind("<Button-1>",clicked)
    draw()
