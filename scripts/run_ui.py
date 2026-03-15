#!/usr/bin/env python3
"""Desktop UI for emotion recognition – embedded preview + theme switch."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Colour palettes ──────────────────────────────────────────────────────────

LIGHT: dict[str, str] = {
    "bg":           "#F0F4F8",
    "panel":        "#FFFFFF",
    "border":       "#E2E8F0",
    "text":         "#1A202C",
    "muted":        "#718096",
    "accent":       "#4F46E5",
    "accent_dim":   "#4338CA",
    "accent_fg":    "#FFFFFF",
    "btn":          "#EDF2F7",
    "btn_hover":    "#DFE6EF",
    "src_on":       "#4F46E5",
    "src_on_fg":    "#FFFFFF",
    "entry_bg":     "#FFFFFF",
    "entry_fg":     "#1A202C",
    "preview_bg":   "#E8ECF2",
    "stop":         "#DC2626",
    "stop_dim":     "#B91C1C",
}

DARK: dict[str, str] = {
    "bg":           "#0D1117",
    "panel":        "#161B22",
    "border":       "#30363D",
    "text":         "#E6EDF3",
    "muted":        "#8B949E",
    "accent":       "#7C3AED",
    "accent_dim":   "#6D28D9",
    "accent_fg":    "#FFFFFF",
    "btn":          "#21262D",
    "btn_hover":    "#30363D",
    "src_on":       "#7C3AED",
    "src_on_fg":    "#FFFFFF",
    "entry_bg":     "#0D1117",
    "entry_fg":     "#E6EDF3",
    "preview_bg":   "#0D1117",
    "stop":         "#DC2626",
    "stop_dim":     "#991B1B",
}

_SOURCES: list[tuple[str, str, str]] = [
    ("image",  "🖼", "Изображение"),
    ("video",  "🎬", "Видео"),
    ("webcam", "📷", "Видеокамера"),
]


class EmotionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("1160x820")
        self.root.minsize(960, 680)

        self._dark = True
        self._p: dict[str, str] = DARK.copy()

        # ── Tk variables
        self.source_type = tk.StringVar(value="image")
        self.input_path = tk.StringVar()
        self.model_path = tk.StringVar(value=self._default_model())
        self.output_path = tk.StringVar()
        self.camera_idx = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Ready")

        # ── Runtime
        self.pipeline: Optional[Any] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.running = False
        self.photo: Optional[ImageTk.PhotoImage] = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._infer_thread: Optional[threading.Thread] = None

        # ── Widget refs
        self.preview_lbl: Optional[tk.Label] = None
        self.start_btn: Optional[tk.Button] = None
        self.theme_btn: Optional[tk.Button] = None
        self.input_entry: Optional[tk.Entry] = None
        self.model_entry: Optional[tk.Entry] = None
        self.output_entry: Optional[tk.Entry] = None
        self.cam_spin: Optional[tk.Spinbox] = None
        self.browse_input_btn: Optional[tk.Button] = None
        self.browse_output_btn: Optional[tk.Button] = None
        self._src_btns: dict[str, tk.Button] = {}
        # Collapsible sidebar blocks (header label + content frame)
        self._block_file: list[tk.Widget] = []
        self._block_cam: list[tk.Widget] = []
        self._block_out: list[tk.Widget] = []

        # ── Theme registries (widget → role)
        self._r_bg: list[tk.Widget] = []        # bg color
        self._r_panel: list[tk.Widget] = []     # panel color (bg)
        self._r_text: list[tk.Label] = []       # primary fg
        self._r_muted: list[tk.Label] = []      # muted fg
        self._r_btn: list[tk.Button] = []       # normal buttons
        self._r_entry: list[tk.Entry] = []      # text entries
        self._r_spin: list[tk.Spinbox] = []     # spinboxes
        self._r_prev: list[tk.Widget] = []      # preview bg

        self._build_ui()
        self._apply_theme()   # apply default dark palette
        self._select_source("image")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _default_model(self) -> str:
        m = (
            Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "best_emotion_model.pt"
        )
        return str(m) if m.exists() else ""

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        p = self._p
        self.root.configure(bg=p["bg"])

        # ── Header bar
        hdr = tk.Frame(self.root, bg=p["panel"], height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self._r_panel.append(hdr)

        title_lbl = tk.Label(
            hdr,
            text="🎭  Emotion Recognition",
            font=("Segoe UI", 14, "bold"),
            bg=p["panel"],
            fg=p["text"],
        )
        title_lbl.pack(side="left", padx=22)
        self._r_panel.append(title_lbl)
        self._r_text.append(title_lbl)

        self.theme_btn = tk.Button(
            hdr,
            text="🌙  Dark",
            font=("Segoe UI", 10),
            bg=p["btn"],
            fg=p["text"],
            activebackground=p["btn_hover"],
            activeforeground=p["text"],
            relief="flat",
            bd=0,
            padx=16,
            pady=7,
            cursor="hand2",
            command=self._toggle_theme,
        )
        self.theme_btn.pack(side="right", padx=18, pady=10)
        self._r_btn.append(self.theme_btn)

        # Thin header separator
        tk.Frame(self.root, bg=p["border"], height=1).pack(fill="x")

        # ── Body
        body = tk.Frame(self.root, bg=p["bg"])
        body.pack(fill="both", expand=True)
        self._r_bg.append(body)

        # Sidebar
        sidebar = tk.Frame(body, bg=p["panel"], width=316)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._r_panel.append(sidebar)
        self._build_sidebar(sidebar)

        # Vertical separator
        tk.Frame(body, bg=p["border"], width=1).pack(side="left", fill="y")

        # Preview column
        right = tk.Frame(body, bg=p["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._r_bg.append(right)
        self._build_preview(right)

    # ── section sub-header helper ─────────────────────────────────────────────

    def _make_sec_label(self, parent: tk.Widget, text: str) -> tk.Label:
        """Create and pack a section label; return it for show/hide."""
        lbl = tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 8, "bold"),
            bg=self._p["panel"],
            fg=self._p["muted"],
            anchor="w",
        )
        lbl.pack(fill="x", padx=20, pady=(16, 3))
        self._r_panel.append(lbl)
        self._r_muted.append(lbl)
        return lbl

    def _sec(self, parent: tk.Widget, text: str) -> None:
        self._make_sec_label(parent, text)

    # ── browse button factory ─────────────────────────────────────────────────

    def _browse_btn(
        self, parent: tk.Widget, command: Any
    ) -> tk.Button:
        p = self._p
        btn = tk.Button(
            parent,
            text="…",
            font=("Segoe UI", 12),
            bg=p["btn"],
            fg=p["text"],
            activebackground=p["btn_hover"],
            activeforeground=p["text"],
            relief="flat",
            bd=0,
            padx=11,
            pady=4,
            cursor="hand2",
            command=command,
        )
        self._r_btn.append(btn)
        return btn

    # ── entry factory ─────────────────────────────────────────────────────────

    def _entry(
        self, parent: tk.Widget, var: tk.Variable
    ) -> tk.Entry:
        p = self._p
        e = tk.Entry(
            parent,
            textvariable=var,
            bg=p["entry_bg"],
            fg=p["entry_fg"],
            insertbackground=p["entry_fg"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=p["border"],
            highlightcolor=p["accent"],
        )
        self._r_entry.append(e)
        return e

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent: tk.Widget) -> None:
        p = self._p

        # ── Source selector buttons
        self._sec(parent, "INPUT SOURCE")

        src_row = tk.Frame(parent, bg=p["panel"])
        src_row.pack(fill="x", padx=16, pady=(2, 0))
        self._r_panel.append(src_row)

        for col, (value, icon, label) in enumerate(_SOURCES):
            btn = tk.Button(
                src_row,
                text=f"{icon}\n{label}",
                font=("Segoe UI", 9, "bold"),
                bg=p["btn"],
                fg=p["text"],
                activebackground=p["btn_hover"],
                activeforeground=p["text"],
                relief="flat",
                bd=0,
                padx=4,
                pady=11,
                cursor="hand2",
                wraplength=84,
                command=lambda v=value: self._select_source(v),
            )
            btn.grid(row=0, column=col, padx=(0, 4), sticky="ew")
            self._src_btns[value] = btn

        src_row.columnconfigure(0, weight=1)
        src_row.columnconfigure(1, weight=1)
        src_row.columnconfigure(2, weight=1)

        # ── Fields container – always packed here, blocks live inside it
        # so pack_forget/pack inside never disturbs the spacer below.
        fields = tk.Frame(parent, bg=p["panel"])
        fields.pack(fill="x")
        self._r_panel.append(fields)

        # ── Input file block
        file_hdr = self._make_sec_label(fields, "INPUT FILE")
        file_body = tk.Frame(fields, bg=p["panel"])
        file_body.pack(fill="x", padx=16, pady=(2, 0))
        self._r_panel.append(file_body)
        self._block_file = [file_hdr, file_body]

        self.input_entry = self._entry(file_body, self.input_path)
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.browse_input_btn = self._browse_btn(
            file_body, command=self._browse_input
        )
        self.browse_input_btn.pack(side="left", padx=(7, 0))

        # ── Camera index block
        cam_hdr = self._make_sec_label(fields, "CAMERA INDEX")
        cam_body = tk.Frame(fields, bg=p["panel"])
        cam_body.pack(fill="x", padx=16, pady=(2, 0))
        self._r_panel.append(cam_body)
        self._block_cam = [cam_hdr, cam_body]

        self.cam_spin = tk.Spinbox(
            cam_body,
            from_=0,
            to=10,
            textvariable=self.camera_idx,
            width=8,
            bg=p["entry_bg"],
            fg=p["entry_fg"],
            insertbackground=p["entry_fg"],
            buttonbackground=p["btn"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=p["border"],
        )
        self.cam_spin.pack(anchor="w", ipady=5)
        self._r_spin.append(self.cam_spin)

        # ── Model checkpoint (always visible)
        self._sec(fields, "MODEL CHECKPOINT")
        m_row = tk.Frame(fields, bg=p["panel"])
        m_row.pack(fill="x", padx=16, pady=(2, 0))
        self._r_panel.append(m_row)

        self.model_entry = self._entry(m_row, self.model_path)
        self.model_entry.pack(
            side="left", fill="x", expand=True, ipady=6
        )
        self._browse_btn(
            m_row, command=self._browse_model
        ).pack(side="left", padx=(7, 0))

        # ── Output video block
        out_hdr = self._make_sec_label(fields, "OUTPUT VIDEO  (OPTIONAL)")
        out_body = tk.Frame(fields, bg=p["panel"])
        out_body.pack(fill="x", padx=16, pady=(2, 0))
        self._r_panel.append(out_body)
        self._block_out = [out_hdr, out_body]

        self.output_entry = self._entry(out_body, self.output_path)
        self.output_entry.pack(
            side="left", fill="x", expand=True, ipady=6
        )
        self.browse_output_btn = self._browse_btn(
            out_body, command=self._browse_output
        )
        self.browse_output_btn.pack(side="left", padx=(7, 0))

        # ── Spacer pushes start button to bottom
        spacer = tk.Frame(parent, bg=p["panel"])
        spacer.pack(fill="both", expand=True)
        self._r_panel.append(spacer)

        # ── Status label (above Start button)
        st_lbl = tk.Label(
            parent,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=p["panel"],
            fg=p["muted"],
            anchor="w",
        )
        st_lbl.pack(fill="x", padx=16, pady=(0, 4))
        self._r_panel.append(st_lbl)
        self._r_muted.append(st_lbl)

        # ── Start / Stop button
        btn_wrap = tk.Frame(parent, bg=p["panel"])
        btn_wrap.pack(fill="x", padx=16, pady=(0, 18))
        self._r_panel.append(btn_wrap)

        self.start_btn = tk.Button(
            btn_wrap,
            text="▶  Start",
            font=("Segoe UI", 11, "bold"),
            bg=p["accent"],
            fg=p["accent_fg"],
            activebackground=p["accent_dim"],
            activeforeground=p["accent_fg"],
            relief="flat",
            bd=0,
            pady=13,
            cursor="hand2",
            command=self._toggle_start_stop,
        )
        self.start_btn.pack(fill="x")

    # ── preview area ──────────────────────────────────────────────────────────

    def _build_preview(self, parent: tk.Widget) -> None:
        p = self._p

        prev_bg = tk.Frame(parent, bg=p["preview_bg"])
        prev_bg.pack(fill="both", expand=True, padx=16, pady=16)
        self._r_prev.append(prev_bg)

        self.preview_lbl = tk.Label(
            prev_bg,
            bg=p["preview_bg"],
            fg=p["muted"],
            text="Preview will appear here",
            font=("Segoe UI", 13),
            anchor="center",
        )
        self.preview_lbl.pack(fill="both", expand=True)
        self._r_prev.append(self.preview_lbl)

    # ── theme ─────────────────────────────────────────────────────────────────

    def _toggle_theme(self) -> None:
        self._dark = not self._dark
        self._p = DARK.copy() if self._dark else LIGHT.copy()
        self._apply_theme()

    def _apply_theme(self) -> None:
        p = self._p
        self.root.configure(bg=p["bg"])

        for w in self._r_bg:
            w.configure(bg=p["bg"])
        for w in self._r_panel:
            w.configure(bg=p["panel"])
        for w in self._r_text:
            w.configure(fg=p["text"])
        for w in self._r_muted:
            w.configure(fg=p["muted"])
        for w in self._r_btn:
            w.configure(
                bg=p["btn"],
                fg=p["text"],
                activebackground=p["btn_hover"],
                activeforeground=p["text"],
            )
        for w in self._r_entry:
            w.configure(
                bg=p["entry_bg"],
                fg=p["entry_fg"],
                insertbackground=p["entry_fg"],
                highlightbackground=p["border"],
                highlightcolor=p["accent"],
            )
        for w in self._r_spin:
            w.configure(
                bg=p["entry_bg"],
                fg=p["entry_fg"],
                insertbackground=p["entry_fg"],
                buttonbackground=p["btn"],
                highlightbackground=p["border"],
            )
        for w in self._r_prev:
            w.configure(bg=p["preview_bg"])
            try:
                w.configure(fg=p["muted"])
            except Exception:
                pass

        # Theme button label
        if self.theme_btn is not None:
            self.theme_btn.configure(
                text="☀️  Light" if self._dark else "🌙  Dark",
                bg=p["btn"],
                fg=p["text"],
                activebackground=p["btn_hover"],
                activeforeground=p["text"],
            )

        # Restore start button accent (only when not streaming)
        if self.start_btn is not None and not self.running:
            self.start_btn.configure(
                bg=p["accent"],
                fg=p["accent_fg"],
                activebackground=p["accent_dim"],
                activeforeground=p["accent_fg"],
            )

        self._refresh_src_btns()

    # ── source selection ──────────────────────────────────────────────────────

    def _select_source(self, value: str) -> None:
        self.source_type.set(value)
        self._refresh_src_btns()
        self._update_field_states()

    def _refresh_src_btns(self) -> None:
        p = self._p
        active = self.source_type.get()

        for key, btn in self._src_btns.items():
            if key == active:
                btn.configure(
                    bg=p["src_on"],
                    fg=p["src_on_fg"],
                    activebackground=p["accent_dim"],
                    activeforeground=p["src_on_fg"],
                )
            else:
                btn.configure(
                    bg=p["btn"],
                    fg=p["text"],
                    activebackground=p["btn_hover"],
                    activeforeground=p["text"],
                )

    def _update_field_states(self) -> None:
        source = self.source_type.get()

        is_image = source == "image"
        is_webcam = source == "webcam"

        input_state = "disabled" if is_webcam else "normal"
        cam_state = "normal" if is_webcam else "disabled"
        out_state = "disabled" if is_image else "normal"

        if self.input_entry is not None:
            self.input_entry.configure(state=input_state)
        if self.browse_input_btn is not None:
            self.browse_input_btn.configure(state=input_state)
        if self.cam_spin is not None:
            self.cam_spin.configure(state=cam_state)
        if self.output_entry is not None:
            self.output_entry.configure(state=out_state)
        if self.browse_output_btn is not None:
            self.browse_output_btn.configure(state=out_state)

    # ── browse dialogs ────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        if self.source_type.get() == "image":
            path = filedialog.askopenfilename(
                title="Select image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All files", "*.*"),
                ],
            )
        if path:
            self.input_path.set(path)

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

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("AVI video", "*.avi"),
            ],
        )
        if path:
            self.output_path.set(path)

    # ── inference control ─────────────────────────────────────────────────────

    def _toggle_start_stop(self) -> None:
        if self.running:
            self._stop_stream("Stopped by user")
            return

        if not self._validate():
            return

        self.status_var.set("Loading model…")
        self.root.update_idletasks()

        model_path = self.model_path.get().strip() or None
        self.pipeline = self._load_pipeline(model_path)
        if self.pipeline is None:
            return

        if self.source_type.get() == "image":
            self._run_image()
        else:
            self._run_stream()

    def _validate(self) -> bool:
        model = self.model_path.get().strip()
        if model and not Path(model).exists():
            messagebox.showerror(
                "Validation error",
                "Selected model checkpoint does not exist.",
            )
            return False

        if self.source_type.get() in {"image", "video"}:
            path = self.input_path.get().strip()
            if not path:
                messagebox.showerror(
                    "Validation error",
                    "Please choose an input file.",
                )
                return False
            if not Path(path).exists():
                messagebox.showerror(
                    "Validation error",
                    "Selected input file does not exist.",
                )
                return False

        return True

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

    def _run_image(self) -> None:
        assert self.pipeline is not None

        img = cv2.imread(self.input_path.get().strip())
        if img is None:
            messagebox.showerror(
                "Runtime error",
                "Cannot read selected image.",
            )
            self.status_var.set("Image read failed")
            return

        results = self.pipeline.process_frame(img)
        out = self.pipeline.draw_results(img, results)
        self._show_frame(out)

        if results:
            labels = ", ".join(r["emotion"] for r in results)
            self.status_var.set(f"Detected: {labels}")
        else:
            self.status_var.set("No face detected")

    def _run_stream(self) -> None:
        src: Union[str, int]
        if self.source_type.get() == "webcam":
            src = int(self.camera_idx.get())
        else:
            src = self.input_path.get().strip()

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror(
                "Runtime error",
                "Cannot open selected source.",
            )
            self.status_var.set("Source open failed")
            return

        # Set large output buffer so reader never waits on encode
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.writer = self._make_writer(
            self.output_path.get().strip(), fps
        )

        # Clear any leftover frames from a previous run
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

        self.running = True
        self._set_ui_running(True)
        self.status_var.set("Streaming…")

        self._infer_thread = threading.Thread(
            target=self._inference_worker,
            args=(fps,),
            daemon=True,
        )
        self._infer_thread.start()
        self.root.after(10, self._poll_frame)

    def _make_writer(
        self, path: str, fps: int
    ) -> Optional[cv2.VideoWriter]:
        if not path or self.cap is None:
            return None
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (w, h))

    # ── background inference thread ───────────────────────────────────────────

    def _inference_worker(self, fps: int) -> None:
        """Read frames, run model, push annotated BGR into queue."""
        frame_interval = 1.0 / max(fps, 1)
        pipeline = self.pipeline
        cap = self.cap
        writer = self.writer

        while self.running and cap is not None and pipeline is not None:
            t0 = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                # Signal end-of-stream
                try:
                    self._frame_q.put(None, timeout=1)
                except queue.Full:
                    pass
                break

            t_read = time.monotonic()

            results = pipeline.process_frame(frame)
            out = pipeline.draw_results(frame, results)

            if writer is not None:
                writer.write(out)

            # --- Skip frames that piled up in the cap buffer while
            # inference was running, so display stays at real-time.
            inference_ms = time.monotonic() - t_read
            stale = int(inference_ms * fps)  # frames accumulated
            for _ in range(min(stale, 8)):
                cap.grab()  # advance without decoding

            # Drop oldest display frame if consumer is too slow
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait(out)
            except queue.Full:
                pass

            # Pace to source FPS when inference is faster than the stream
            elapsed = time.monotonic() - t0
            spare = frame_interval - elapsed
            if spare > 0:
                time.sleep(spare)

    # ── main-thread display loop ──────────────────────────────────────────────

    def _poll_frame(self) -> None:
        """Called by Tk event loop – just dequeues and displays a frame."""
        if not self.running:
            return

        try:
            frame = self._frame_q.get_nowait()
        except queue.Empty:
            # No frame ready yet; come back soon
            self.root.after(5, self._poll_frame)
            return

        if frame is None:
            # End-of-stream sentinel from worker
            self._stop_stream("Stream ended")
            return

        self._show_frame(frame)
        # Schedule next poll quickly; Tk itself won't block
        self.root.after(1, self._poll_frame)

    def _show_frame(self, bgr: Any) -> None:
        if self.preview_lbl is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        max_w = max(self.preview_lbl.winfo_width() - 4, 320)
        max_h = max(self.preview_lbl.winfo_height() - 4, 240)
        scale = min(max_w / fw, max_h / fh)
        nw = max(int(fw * scale), 1)
        nh = max(int(fh * scale), 1)

        img = Image.fromarray(
            cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        )
        self.photo = ImageTk.PhotoImage(image=img)
        self.preview_lbl.configure(image=self.photo, text="")

    def _set_ui_running(self, running: bool) -> None:
        p = self._p
        if self.start_btn is not None:
            if running:
                self.start_btn.configure(
                    text="⏹  Stop",
                    bg=p["stop"],
                    activebackground=p["stop_dim"],
                    fg="#FFFFFF",
                    activeforeground="#FFFFFF",
                )
            else:
                self.start_btn.configure(
                    text="▶  Start",
                    bg=p["accent"],
                    activebackground=p["accent_dim"],
                    fg=p["accent_fg"],
                    activeforeground=p["accent_fg"],
                )

        state = "disabled" if running else "normal"
        for w in [
            self.input_entry,
            self.model_entry,
            self.output_entry,
            self.cam_spin,
            self.browse_input_btn,
            self.browse_output_btn,
            *self._src_btns.values(),
        ]:
            if w is not None:
                try:
                    w.configure(state=state)
                except Exception:
                    pass

        if not running:
            self._refresh_src_btns()
            self._update_field_states()

    def _stop_stream(self, reason: str) -> None:
        self.running = False

        # Let the inference thread finish its current iteration
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2)
            self._infer_thread = None

        # Drain queue so the worker can unblock any put()
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self._set_ui_running(False)
        self.status_var.set(reason)

        # Clear preview
        self.photo = None
        if self.preview_lbl is not None:
            self.preview_lbl.configure(image="", text="")

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
