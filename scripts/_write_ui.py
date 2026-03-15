"""Helper: writes run_ui.py with proper UTF-8 encoding."""
import pathlib

TARGET = pathlib.Path(__file__).parent / "run_ui.py"

CONTENT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emotion AI – merged dashboard UI.

Visual style: dashboard (dark panel, overlay, right summary + timeline).
Functionality: image / video / webcam, model path, camera index,
output video, JSON speech data, light/dark theme toggle.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Palettes ──────────────────────────────────────────────────────────────────

DARK: dict[str, str] = {
    "bg":         "#1A1E26",
    "panel":      "#232731",
    "video":      "#12161D",
    "border":     "#353C49",
    "text":       "#E8ECF4",
    "muted":      "#A5ADBB",
    "accent":     "#B443C6",
    "title":      "#FFB100",
    "btn":        "#2D3441",
    "btn_hover":  "#3B4454",
    "btn_active": "#466082",
    "danger":     "#8D2C2C",
    "danger_h":   "#A63838",
    "overlay":    "#102848",
    "entry_bg":   "#1E232D",
    "entry_fg":   "#E8ECF4",
    "tree_bg":    "#1E232D",
    "tree_fg":    "#D3DBE8",
    "tree_head":  "#2B3340",
    "tree_sel":   "#42526A",
}

LIGHT: dict[str, str] = {
    "bg":         "#F0F4F8",
    "panel":      "#FFFFFF",
    "video":      "#E8ECF2",
    "border":     "#CBD5E0",
    "text":       "#1A202C",
    "muted":      "#718096",
    "accent":     "#7C3AED",
    "title":      "#92400E",
    "btn":        "#EDF2F7",
    "btn_hover":  "#DFE6EF",
    "btn_active": "#C7D2FE",
    "danger":     "#DC2626",
    "danger_h":   "#B91C1C",
    "overlay":    "#1E3A5F",
    "entry_bg":   "#FFFFFF",
    "entry_fg":   "#1A202C",
    "tree_bg":    "#F7F8FC",
    "tree_fg":    "#1A202C",
    "tree_head":  "#E8ECF4",
    "tree_sel":   "#C7D2FE",
}
''' + \
'''

class EmotionApp:
    """Dashboard UI: visual from dashboard sketch + full source/model controls."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Emotion AI Live Analyzer")
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        # theme state
        self._dark = True
        self._p: dict[str, str] = DARK.copy()

        # variables
        self.source_type  = tk.StringVar(value="webcam")
        self.input_path   = tk.StringVar(value="")
        self.model_path   = tk.StringVar(value=self._default_model())
        self.output_path  = tk.StringVar(value="")
        self.camera_idx   = tk.IntVar(value=0)

        self.status_var        = tk.StringVar(value="Ready")
        self.summary_title_var = tk.StringVar(value="Final emotion: NEUTRAL")
        self.source_var        = tk.StringVar(value="Source: camera")
        self.face_var          = tk.StringVar(value="Face emotion: NEUTRAL (0.00)")
        self.speech_var        = tk.StringVar(value="Speech: NEUTRAL (0.93)")
        self.toxicity_var      = tk.StringVar(value="Toxicity: NEUTRAL (0.0)")
        self.text_var          = tk.StringVar(value="Text: ~")

        # internal
        self.speech_sentiment  = "NEUTRAL"
        self.speech_confidence = 0.93
        self.toxicity_score    = 0.0
        self.overlay_face      = "NEUTRAL"

        self.pipeline: Optional[Any]           = None
        self.cap:      Optional[cv2.VideoCapture] = None
        self.writer:   Optional[cv2.VideoWriter]  = None
        self.running   = False
        self.photo:    Optional[ImageTk.PhotoImage] = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._infer_thread: Optional[threading.Thread] = None

        # widget refs
        self.preview_lbl:    Optional[tk.Label]     = None
        self.overlay_lbl:    Optional[tk.Label]     = None
        self.frame_time_lbl: Optional[tk.Label]     = None
        self.timeline:       Optional[ttk.Treeview] = None
        self.btn_camera:     Optional[tk.Button]    = None
        self.btn_video:      Optional[tk.Button]    = None
        self.btn_image:      Optional[tk.Button]    = None
        self.btn_json:       Optional[tk.Button]    = None
        self.btn_stop:       Optional[tk.Button]    = None
        self.btn_settings:   Optional[tk.Button]    = None
        self.btn_theme:      Optional[tk.Button]    = None
        self._summary_lbl:   Optional[tk.Label]     = None

        # theme registries
        self._r_bg:    list[tk.Widget] = []
        self._r_panel: list[tk.Widget] = []
        self._r_text:  list[tk.Widget] = []
        self._r_muted: list[tk.Widget] = []
        self._r_entry: list[tk.Entry]  = []

        self._row_id = 0

        self._build_ui()
        self._apply_theme()
        self._set_mode_button_state()
        self._refresh_overlay()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _default_model(self) -> str:
        p = (
            Path(__file__).resolve().parent.parent
            / "checkpoints" / "best_emotion_model.pt"
        )
        return str(p) if p.exists() else ""

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        p = self._p

        self._accent_strip = tk.Frame(self.root, bg=p["accent"], height=4)
        self._accent_strip.pack(fill="x")

        self._topbar = tk.Frame(self.root, bg=p["panel"], height=48)
        self._topbar.pack(fill="x")
        self._topbar.pack_propagate(False)
        self._r_panel.append(self._topbar)

        title_lbl = tk.Label(
            self._topbar,
            text="Emotion AI Live Analyzer",
            bg=p["panel"], fg=p["muted"],
            font=("Segoe UI", 10, "bold"),
        )
        title_lbl.pack(side="left", padx=(12, 20))
        self._r_panel.append(title_lbl)
        self._r_muted.append(title_lbl)

        # left source controls
        src_grp = tk.Frame(self._topbar, bg=p["panel"])
        src_grp.pack(side="left", fill="y", pady=8)
        self._r_panel.append(src_grp)

        self.btn_camera = self._mkbtn(
            src_grp, "Camera", self._on_camera_clicked, 14)
        self.btn_camera.pack(side="left", padx=4)

        self.btn_video = self._mkbtn(
            src_grp, "Load Video", self._on_video_clicked, 16)
        self.btn_video.pack(side="left", padx=4)

        self.btn_image = self._mkbtn(
            src_grp, "Open Image", self._on_image_clicked, 16)
        self.btn_image.pack(side="left", padx=4)

        self.btn_json = self._mkbtn(
            src_grp, "Open JSON", self._on_open_json, 14)
        self.btn_json.pack(side="left", padx=4)

        self.btn_stop = self._mkbtn(
            src_grp, "Stop", self._on_stop_clicked, 10,
            bg=p["danger"], hover=p["danger_h"])
        self.btn_stop.pack(side="left", padx=(4, 0))

        # right controls
        rgt_grp = tk.Frame(self._topbar, bg=p["panel"])
        rgt_grp.pack(side="right", fill="y", pady=8, padx=8)
        self._r_panel.append(rgt_grp)

        self.btn_settings = self._mkbtn(
            rgt_grp, "Settings", self._open_settings, 14)
        self.btn_settings.pack(side="left", padx=4)

        self.btn_theme = self._mkbtn(
            rgt_grp, "Light", self._toggle_theme, 12)
        self.btn_theme.pack(side="left", padx=4)

        self._sep_line = tk.Frame(self.root, bg=p["border"], height=1)
        self._sep_line.pack(fill="x")

        self._content = tk.Frame(self.root, bg=p["bg"])
        self._content.pack(fill="both", expand=True)
        self._r_bg.append(self._content)

        left = tk.Frame(self._content, bg=p["bg"])
        left.pack(side="left", fill="both", expand=True)
        self._r_bg.append(left)

        right = tk.Frame(self._content, bg=p["panel"], width=420)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)
        self._r_panel.append(right)

        self._build_video_panel(left)
        self._build_right_panel(right)

    def _mkbtn(
        self,
        parent: tk.Widget,
        text: str,
        command: Any,
        width: int,
        bg: Optional[str] = None,
        hover: Optional[str] = None,
    ) -> tk.Button:
        p = self._p
        base = bg or p["btn"]
        over = hover or p["btn_hover"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            font=("Segoe UI", 9),
            bg=base, fg=p["text"],
            relief="flat", bd=0,
            padx=6, pady=4,
            activebackground=over,
            activeforeground=p["text"],
            cursor="hand2",
        )

    # ── video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self, parent: tk.Widget) -> None:
        p = self._p

        self._video_shell = tk.Frame(
            parent, bg=p["video"],
            highlightthickness=1,
            highlightbackground=p["border"],
        )
        self._video_shell.pack(
            fill="both", expand=True, padx=(8, 4), pady=8
        )

        self.preview_lbl = tk.Label(
            self._video_shell,
            bg=p["video"], fg=p["muted"],
            text="Waiting for stream",
            font=("Segoe UI", 14),
        )
        self.preview_lbl.pack(fill="both", expand=True)

        self.overlay_lbl = tk.Label(
            self._video_shell,
            bg=p["overlay"], fg="#FFFFFF",
            font=("Segoe UI", 22, "bold"),
            padx=16, pady=8, anchor="w",
        )
        self.overlay_lbl.place(x=16, y=16)

        self._footer = tk.Frame(parent, bg=p["panel"], height=28)
        self._footer.pack(fill="x", padx=(8, 4), pady=(0, 8))
        self._footer.pack_propagate(False)
        self._r_panel.append(self._footer)

        self.frame_time_lbl = tk.Label(
            self._footer,
            text="Last frame: -- ms",
            bg=p["panel"], fg=p["muted"],
            font=("Segoe UI", 9), anchor="w",
        )
        self.frame_time_lbl.pack(fill="x", padx=8)
        self._r_panel.append(self.frame_time_lbl)
        self._r_muted.append(self.frame_time_lbl)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: tk.Widget) -> None:
        p = self._p

        top = tk.Frame(parent, bg=p["panel"])
        top.pack(fill="x")
        self._r_panel.append(top)

        self._summary_lbl = tk.Label(
            top,
            textvariable=self.summary_title_var,
            bg=p["panel"], fg=p["title"],
            font=("Segoe UI", 18, "bold"),
            anchor="w", pady=10,
        )
        self._summary_lbl.pack(fill="x", padx=10)
        self._r_panel.append(self._summary_lbl)

        for var in [
            self.source_var, self.face_var,
            self.speech_var, self.toxicity_var, self.text_var,
        ]:
            lbl = tk.Label(
                top,
                textvariable=var,
                bg=p["panel"], fg=p["text"],
                anchor="w", justify="left",
                font=("Segoe UI", 10),
            )
            lbl.pack(fill="x", padx=10)
            self._r_panel.append(lbl)
            self._r_text.append(lbl)

        status_lbl = tk.Label(
            top,
            textvariable=self.status_var,
            bg=p["panel"], fg=p["muted"],
            anchor="w", font=("Segoe UI", 9), pady=4,
        )
        status_lbl.pack(fill="x", padx=10)
        self._r_panel.append(status_lbl)
        self._r_muted.append(status_lbl)

        self._table_sep = tk.Frame(parent, bg=p["accent"], height=1)
        self._table_sep.pack(fill="x")

        tbl_wrap = tk.Frame(parent, bg=p["panel"])
        tbl_wrap.pack(fill="both", expand=True)
        self._r_panel.append(tbl_wrap)

        self._configure_tree_style()
        cols = ("t", "face", "speech", "final")
        self.timeline = ttk.Treeview(
            tbl_wrap, columns=cols,
            show="headings", style="Dashboard.Treeview",
        )
        for col, heading, w in [
            ("t",      "t, ms",  82),
            ("face",   "Face",   95),
            ("speech", "Speech", 95),
            ("final",  "Final", 100),
        ]:
            self.timeline.heading(col, text=heading)
            self.timeline.column(
                col, width=w, minwidth=w - 15, anchor="center"
            )

        scroll = ttk.Scrollbar(
            tbl_wrap, orient="vertical",
            command=self.timeline.yview,
        )
        self.timeline.configure(yscrollcommand=scroll.set)
        self.timeline.pack(
            side="left", fill="both", expand=True,
            padx=(5, 0), pady=5,
        )
        scroll.pack(side="right", fill="y", pady=5, padx=(0, 5))

    def _configure_tree_style(self) -> None:
        p = self._p
        s = ttk.Style(self.root)
        s.theme_use("default")
        s.configure(
            "Dashboard.Treeview",
            background=p["tree_bg"],
            foreground=p["tree_fg"],
            fieldbackground=p["tree_bg"],
            bordercolor=p["border"],
            rowheight=22,
            font=("Segoe UI", 9),
        )
        s.configure(
            "Dashboard.Treeview.Heading",
            background=p["tree_head"],
            foreground=p["text"],
            bordercolor=p["border"],
            font=("Segoe UI", 9, "bold"),
        )
        s.map(
            "Dashboard.Treeview",
            background=[("selected", p["tree_sel"])],
        )

    # ── theme ─────────────────────────────────────────────────────────────────

    def _toggle_theme(self) -> None:
        self._dark = not self._dark
        self._p = DARK.copy() if self._dark else LIGHT.copy()
        self._apply_theme()

    def _apply_theme(self) -> None:
        p = self._p
        self.root.configure(bg=p["bg"])

        self._accent_strip.configure(bg=p["accent"])
        self._sep_line.configure(bg=p["border"])
        self._table_sep.configure(bg=p["accent"])
        self._video_shell.configure(
            bg=p["video"], highlightbackground=p["border"]
        )
        if self.preview_lbl is not None:
            self.preview_lbl.configure(bg=p["video"], fg=p["muted"])
        if self.overlay_lbl is not None:
            self.overlay_lbl.configure(bg=p["overlay"])
        if self._summary_lbl is not None:
            self._summary_lbl.configure(bg=p["panel"], fg=p["title"])

        for w in self._r_bg:
            w.configure(bg=p["bg"])
        for w in self._r_panel:
            w.configure(bg=p["panel"])
        for w in self._r_text:
            w.configure(fg=p["text"])
        for w in self._r_muted:
            w.configure(fg=p["muted"])
        for e in self._r_entry:
            e.configure(
                bg=p["entry_bg"], fg=p["entry_fg"],
                insertbackground=p["entry_fg"],
                highlightbackground=p["border"],
            )

        if self.btn_theme is not None:
            lbl = "Dark" if not self._dark else "Light"
            self.btn_theme.configure(
                text=lbl, bg=p["btn"], fg=p["text"],
                activebackground=p["btn_hover"],
                activeforeground=p["text"],
            )

        for btn in [
            self.btn_camera, self.btn_video, self.btn_image,
            self.btn_json, self.btn_settings,
        ]:
            if btn is not None:
                btn.configure(
                    bg=p["btn"], fg=p["text"],
                    activebackground=p["btn_hover"],
                    activeforeground=p["text"],
                )
        if self.btn_stop is not None:
            self.btn_stop.configure(
                bg=p["danger"], fg=p["text"],
                activebackground=p["danger_h"],
                activeforeground=p["text"],
            )

        self._configure_tree_style()
        self._set_mode_button_state()

    # ── source highlight ──────────────────────────────────────────────────────

    def _set_mode_button_state(self) -> None:
        p = self._p
        active = self.source_type.get()
        mapping = {
            "webcam": self.btn_camera,
            "video":  self.btn_video,
            "image":  self.btn_image,
        }
        for key, btn in mapping.items():
            if btn is None:
                continue
            btn.configure(
                bg=p["btn_active"] if key == active else p["btn"]
            )

    # ── settings dialog ───────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        p = self._p
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.configure(bg=p["panel"])
        dlg.resizable(False, False)
        dlg.grab_set()

        def lbl_row(text: str) -> tk.Frame:
            tk.Label(
                dlg, text=text,
                bg=p["panel"], fg=p["muted"],
                font=("Segoe UI", 9, "bold"), anchor="w",
            ).pack(fill="x", padx=18, pady=(12, 2))
            f = tk.Frame(dlg, bg=p["panel"])
            f.pack(fill="x", padx=14, pady=(0, 2))
            return f

        def mk_entry(parent: tk.Widget, var: tk.StringVar) -> tk.Entry:
            e = tk.Entry(
                parent, textvariable=var,
                bg=p["entry_bg"], fg=p["entry_fg"],
                insertbackground=p["entry_fg"],
                relief="flat", bd=0,
                highlightthickness=1,
                highlightbackground=p["border"],
                highlightcolor=p["accent"],
                font=("Segoe UI", 9),
            )
            self._r_entry.append(e)
            return e

        def mk_browse(parent: tk.Widget, cmd: Any) -> tk.Button:
            return tk.Button(
                parent, text="...", command=cmd,
                bg=p["btn"], fg=p["text"],
                activebackground=p["btn_hover"],
                activeforeground=p["text"],
                relief="flat", bd=0,
                padx=10, pady=4,
                font=("Segoe UI", 10),
                cursor="hand2",
            )

        # Model checkpoint
        r = lbl_row("Model checkpoint")
        mk_entry(r, self.model_path).pack(
            side="left", fill="x", expand=True, ipady=5
        )
        mk_browse(r, self._browse_model).pack(side="left", padx=(6, 0))

        # Camera index
        r2 = lbl_row("Camera index")
        tk.Spinbox(
            r2,
            from_=0, to=10,
            textvariable=self.camera_idx,
            width=8,
            bg=p["entry_bg"], fg=p["entry_fg"],
            insertbackground=p["entry_fg"],
            buttonbackground=p["btn"],
            relief="flat", bd=0,
            highlightthickness=1,
            highlightbackground=p["border"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", ipady=5)

        # Output video
        r3 = lbl_row("Output video (optional)")
        mk_entry(r3, self.output_path).pack(
            side="left", fill="x", expand=True, ipady=5
        )
        mk_browse(r3, self._browse_output).pack(side="left", padx=(6, 0))

        tk.Button(
            dlg, text="Close", command=dlg.destroy,
            bg=p["accent"], fg="#FFFFFF",
            activebackground=p["btn_hover"],
            activeforeground="#FFFFFF",
            relief="flat", bd=0,
            padx=20, pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).pack(pady=(18, 16))

        self.root.update_idletasks()
        dlg.update_idletasks()
        px = self.root.winfo_x() + self.root.winfo_width() // 2
        py = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.geometry(
            "+%d+%d" % (px - dlg.winfo_width() // 2,
                        py - dlg.winfo_height() // 2)
        )

    # ── browse helpers ────────────────────────────────────────────────────────

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[
                ("PyTorch checkpoint", "*.pt *.pth"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.model_path.set(path)
            self.pipeline = None  # force reload

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
        )
        if path:
            self.output_path.set(path)

    # ── source handlers ───────────────────────────────────────────────────────

    def _on_camera_clicked(self) -> None:
        self.source_type.set("webcam")
        self.input_path.set("")
        self._set_mode_button_state()
        self._start_stream()

    def _on_video_clicked(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.source_type.set("video")
        self.input_path.set(path)
        self._set_mode_button_state()
        self._start_stream()

    def _on_image_clicked(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if self.running:
            self._stop_stream("Restarting with image")
        self.source_type.set("image")
        self.input_path.set(path)
        self._set_mode_button_state()
        self._run_image(path)

    def _on_open_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Open JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8")
            )
        except Exception as exc:
            messagebox.showerror("JSON error", str(exc))
            return

        self.speech_sentiment = str(
            payload.get("speech_sentiment", "NEUTRAL")
        ).upper()
        self.speech_confidence = float(
            payload.get("speech_confidence", 0.93)
        )
        self.toxicity_score = float(payload.get("toxicity", 0.0))
        tox_lbl = "TOXIC" if self.toxicity_score >= 0.6 else "NEUTRAL"
        raw = str(payload.get("text", "~")).strip() or "~"
        text_val = raw[:117] + "..." if len(raw) > 120 else raw

        self.speech_var.set(
            f"Speech: {self.speech_sentiment}"
            f" ({self.speech_confidence:.2f})"
        )
        self.toxicity_var.set(
            f"Toxicity: {tox_lbl} ({self.toxicity_score:.1f})"
        )
        self.text_var.set(f"Text: {text_val}")
        self.status_var.set(f"JSON loaded: {Path(path).name}")
        self._refresh_overlay()

    def _on_stop_clicked(self) -> None:
        if self.running:
            self._stop_stream("Stopped by user")

    # ── single image inference ────────────────────────────────────────────────

    def _run_image(self, image_path: str) -> None:
        if not self._ensure_pipeline():
            return
        assert self.pipeline is not None

        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror(
                "Error", "Cannot read selected image."
            )
            self.status_var.set("Image read failed")
            return

        self.status_var.set("Analysing image...")
        self.root.update_idletasks()

        results = self.pipeline.process_frame(img)
        out = self.pipeline.draw_results(img, results)
        self._show_frame(out)

        ts = int(time.time() * 1000)
        self._update_dashboard(results, ts)
        self.source_var.set(
            f"Source: image ({Path(image_path).name})"
        )

        if results:
            names = ", ".join(r["emotion"].upper() for r in results)
            self.status_var.set(f"Detected: {names}")
        else:
            self.status_var.set("No faces detected")

    # ── pipeline ──────────────────────────────────────────────────────────────

    def _ensure_pipeline(self) -> bool:
        if self.pipeline is not None:
            return True
        self.status_var.set("Loading model...")
        self.root.update_idletasks()
        mp = self.model_path.get().strip() or None
        self.pipeline = self._load_pipeline(mp)
        return self.pipeline is not None

    def _load_pipeline(
        self, model_path: Optional[str]
    ) -> Optional[Any]:
        try:
            from src.pipeline import EmotionPipeline
            return EmotionPipeline(
                model_checkpoint=model_path,
                face_confidence=0.85,
                input_size=112,
            )
        except Exception as exc:
            messagebox.showerror("Runtime error", str(exc))
            self.status_var.set("Model load failed")
            return None

    # ── streaming ─────────────────────────────────────────────────────────────

    def _start_stream(self) -> None:
        if self.running:
            self._stop_stream("Restarting")

        if not self._ensure_pipeline():
            return

        if self.source_type.get() == "webcam":
            source: Union[int, str] = int(self.camera_idx.get())
            self.source_var.set("Source: camera")
        else:
            source = self.input_path.get().strip()
            if not source or not Path(str(source)).exists():
                messagebox.showerror("Error", "Video file not found.")
                return
            self.source_var.set(
                f"Source: video ({Path(str(source)).name})"
            )

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror(
                "Error", "Cannot open video source."
            )
            self.status_var.set("Source open failed")
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.writer = self._make_writer(
            self.output_path.get().strip(), fps
        )

        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

        self.running = True
        self._toggle_stream_ui(True)
        self.status_var.set("Streaming...")

        self._infer_thread = threading.Thread(
            target=self._inference_worker, args=(fps,), daemon=True
        )
        self._infer_thread.start()
        self.root.after(1, self._poll_frame)

    def _make_writer(
        self, path: str, fps: int
    ) -> Optional[cv2.VideoWriter]:
        if not path or self.cap is None:
            return None
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (w, h))

    def _toggle_stream_ui(self, streaming: bool) -> None:
        state = "disabled" if streaming else "normal"
        for btn in [
            self.btn_camera, self.btn_video,
            self.btn_image, self.btn_json,
        ]:
            if btn is not None:
                btn.configure(state=state)
        if self.btn_stop is not None:
            self.btn_stop.configure(
                state="normal" if streaming else "disabled"
            )
        if not streaming:
            self._set_mode_button_state()

    # ── inference thread ──────────────────────────────────────────────────────

    def _inference_worker(self, fps: int) -> None:
        interval = 1.0 / max(fps, 1)
        pipeline = self.pipeline
        cap = self.cap
        writer = self.writer

        while self.running and cap is not None and pipeline is not None:
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                try:
                    self._frame_q.put(None, timeout=1)
                except queue.Full:
                    pass
                break

            ts_ms = int(time.time() * 1000)
            results = pipeline.process_frame(frame)
            out = pipeline.draw_results(frame, results)

            if writer is not None:
                writer.write(out)

            stale = int((time.monotonic() - t0) * fps)
            for _ in range(min(stale, 8)):
                cap.grab()

            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait((out, results, ts_ms))
            except queue.Full:
                pass

            spare = interval - (time.monotonic() - t0)
            if spare > 0:
                time.sleep(spare)

    # ── display loop ──────────────────────────────────────────────────────────

    def _poll_frame(self) -> None:
        if not self.running:
            return
        try:
            item = self._frame_q.get_nowait()
        except queue.Empty:
            self.root.after(5, self._poll_frame)
            return
        if item is None:
            self._stop_stream("Stream ended")
            return
        frame, results, ts_ms = item
        self._show_frame(frame)
        self._update_dashboard(results, ts_ms)
        self.root.after(1, self._poll_frame)

    def _show_frame(self, bgr: Any) -> None:
        if self.preview_lbl is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        max_w = max(self.preview_lbl.winfo_width() - 4, 320)
        max_h = max(self.preview_lbl.winfo_height() - 4, 240)
        scale = min(max_w / fw, max_h / fh)
        nw, nh = max(int(fw * scale), 1), max(int(fh * scale), 1)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.preview_lbl.configure(image=self.photo, text="")

    # ── dashboard update ──────────────────────────────────────────────────────

    def _update_dashboard(
        self, results: list[dict], ts_ms: int
    ) -> None:
        if results:
            best = max(
                results,
                key=lambda r: float(r.get("confidence", 0.0)),
            )
            self.overlay_face = str(
                best.get("emotion", "neutral")
            ).upper()
            conf = float(best.get("confidence", 0.0))
            self.face_var.set(
                f"Face emotion: {self.overlay_face} ({conf:.2f})"
            )
        else:
            self.overlay_face = "NEUTRAL"
            self.face_var.set("Face emotion: NEUTRAL (0.00)")

        self.speech_var.set(
            f"Speech: {self.speech_sentiment}"
            f" ({self.speech_confidence:.2f})"
        )
        tox_lbl = "TOXIC" if self.toxicity_score >= 0.6 else "NEUTRAL"
        self.toxicity_var.set(
            f"Toxicity: {tox_lbl} ({self.toxicity_score:.1f})"
        )

        final = self._compose_final(
            self.overlay_face, self.speech_sentiment
        )
        self.summary_title_var.set(f"Final emotion: {final}")
        self._refresh_overlay()

        if self.frame_time_lbl is not None:
            self.frame_time_lbl.configure(
                text=f"Last frame: {ts_ms} ms"
            )

        if self.timeline is not None:
            self._row_id += 1
            self.timeline.insert(
                "", "end", iid=str(self._row_id),
                values=(
                    ts_ms, self.overlay_face,
                    self.speech_sentiment, final,
                ),
            )
            children = self.timeline.get_children()
            if len(children) > 250:
                self.timeline.delete(children[0])
            self.timeline.yview_moveto(1.0)

    def _compose_final(self, face: str, speech: str) -> str:
        fu, su = face.upper(), speech.upper()
        if fu == su:
            return fu
        if su == "NEUTRAL":
            return fu
        if fu == "NEUTRAL":
            return su
        return fu

    def _refresh_overlay(self) -> None:
        if self.overlay_lbl is None:
            return
        final = (
            self.summary_title_var.get()
            .split(":", maxsplit=1)[-1]
            .strip()
            .upper()
        )
        self.overlay_lbl.configure(
            text=(
                f"{final} \\u00b7 face {self.overlay_face}"
                f" \\u00b7 voice {self.speech_sentiment}"
            )
        )

    # ── cleanup ───────────────────────────────────────────────────────────────

    def _stop_stream(self, reason: str) -> None:
        self.running = False
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2)
            self._infer_thread = None
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._toggle_stream_ui(False)
        self.status_var.set(reason)
        self.photo = None
        if self.preview_lbl is not None:
            self.preview_lbl.configure(image="", text="Waiting for stream")

    def _on_close(self) -> None:
        if self.running:
            self._stop_stream("Stopped")
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    EmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
'''

TARGET.write_text(CONTENT, encoding="utf-8")
print("Written", TARGET, TARGET.stat().st_size, "bytes")
