#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import sounddevice as sd
except Exception:
    sd = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.speech_to_text import (
    TranscriptSegment,
    VoskAudioFileRecognizer,
    VoskMicrophoneRecognizer,
    extract_audio_track,
)
from src.audio.text_emotion import RuBertTextEmotionAnalyzer
from src.pipeline import EmotionPipeline


@dataclass
class AnalyzerState:
    emotion: str = "NEUTRAL"
    face_confidence: float = 0.0
    voice_confidence: float = 0.0
    subtitle: str = "Ready"
    camera_enabled: bool = False
    mic_enabled: bool = False
    camera_index: int = 0
    mic_device: Optional[int] = None
    show_subtitles: bool = True
    show_video_emotion_panel: bool = True


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str
    text_emotion: str
    confidence: float
    color: str


class GlassPanel(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(42)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)


class RoundButton(QPushButton):
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setFixedSize(52, 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class WindowButton(QPushButton):
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setFixedSize(30, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class SettingsDialog(QDialog):
    def __init__(
        self,
        camera_items: list[tuple[str, int]],
        microphone_items: list[tuple[str, Optional[int]]],
        current_camera: int,
        current_microphone: Optional[int],
        show_subtitles: bool,
        show_video_emotion_panel: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setObjectName("SettingsDialog")
        self.setMinimumWidth(460)

        self.camera_combo = QComboBox(self)
        self.microphone_combo = QComboBox(self)
        self.subtitle_checkbox = QCheckBox("Show subtitles", self)
        self.video_emotion_panel_checkbox = QCheckBox("Show video emotion panel", self)
        self.subtitle_checkbox.setChecked(show_subtitles)
        self.video_emotion_panel_checkbox.setChecked(show_video_emotion_panel)

        for label, value in camera_items:
            self.camera_combo.addItem(label, value)
        for label, value in microphone_items:
            self.microphone_combo.addItem(label, value)

        self._select_value(self.camera_combo, current_camera)
        self._select_value(self.microphone_combo, current_microphone)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("Device and subtitle settings", self)
        header.setObjectName("SettingsHeader")
        layout.addWidget(header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.addRow("Camera", self.camera_combo)
        form.addRow("Microphone", self.microphone_combo)
        layout.addLayout(form)

        layout.addWidget(self.subtitle_checkbox)
        layout.addWidget(self.video_emotion_panel_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_value(self, combo: QComboBox, value: Optional[int]) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def selected_camera(self) -> int:
        return int(self.camera_combo.currentData())

    def selected_microphone(self) -> Optional[int]:
        data = self.microphone_combo.currentData()
        return None if data is None else int(data)

    def subtitles_enabled(self) -> bool:
        return self.subtitle_checkbox.isChecked()

    def video_emotion_panel_enabled(self) -> bool:
        return self.video_emotion_panel_checkbox.isChecked()


class BottomControls(GlassPanel):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        camera_enabled: bool = False,
        mic_enabled: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("BottomControls")
        self.setFixedSize(92, 248)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(18)

        self.camera_button = RoundButton("📷")
        self.mic_button = RoundButton("🎤")
        self.video_button = RoundButton("🎬")
        self.settings_button = RoundButton("⚙")
        self.camera_button.setCheckable(True)
        self.mic_button.setCheckable(True)
        self.video_button.setCheckable(False)
        self.settings_button.setCheckable(False)
        self.camera_button.setChecked(camera_enabled)
        self.mic_button.setChecked(mic_enabled)
        self.mic_button.setVisible(False)

        layout.addStretch(1)
        layout.addWidget(self.camera_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        self._refresh_button_style()

    def _refresh_button_style(self) -> None:
        self.camera_button.setProperty("active", self.camera_button.isChecked())
        self.mic_button.setProperty("active", self.mic_button.isChecked())
        self.settings_button.setProperty("active", self.settings_button.isChecked())
        for button in (self.camera_button, self.mic_button, self.video_button, self.settings_button):
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()


class PlayerControls(GlassPanel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlayerControls")
        self.setFixedSize(180, 72)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        self.play_pause_button = RoundButton("⏸")
        self.restart_button = RoundButton("↺")
        self.play_pause_button.setFixedSize(50, 50)
        self.restart_button.setFixedSize(50, 50)

        layout.addWidget(self.play_pause_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.restart_button, alignment=Qt.AlignmentFlag.AlignCenter)


class EmotionAILiveAnalyzer(QMainWindow):
    voice_partial = pyqtSignal(str)
    voice_final = pyqtSignal(str)
    voice_error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Emotion AI Live Analyzer")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMinimumSize(1180, 760)
        self.setMouseTracking(True)

        self.state = AnalyzerState()
        self._drag_offset: Optional[QPoint] = None
        self._current_frame: Optional[np.ndarray] = None
        self._camera: Optional[cv2.VideoCapture] = None
        self._pipeline: Optional[EmotionPipeline] = None
        self._voice_recognizer: Optional[VoskMicrophoneRecognizer] = None
        self._text_emotion_analyzer: Optional[RuBertTextEmotionAnalyzer] = None
        self._rubert_init_error: Optional[str] = None
        self._live_text_emotion = "NEUTRAL"
        self._live_text_confidence = 0.0
        self._voice_model_path = self._default_vosk_model_path()
        self._rubert_model_path = self._default_rubert_model_path()
        self._analysis_loading = False
        self._source_mode = "camera"
        self._video_path: Optional[str] = None
        self._video_paused = False
        self._video_ended = False
        self._last_frame: Optional[np.ndarray] = None
        self._cached_results: list[dict] = []
        self._last_video_results: list[dict] = []
        self._video_face_results_by_frame: dict[int, list[dict]] = {}
        self._video_subtitle_cues: list[SubtitleCue] = []
        self._active_video_cue_index = -1
        self._video_fps = 30.0
        self._video_total_frames = 0
        self._video_frame_index = 0
        self._video_analyze_every_n = 2
        self._camera_timer_interval_ms = 33
        self._video_timer_interval_ms = 33
        self._analysis_busy = False
        self._analysis_lock = threading.Lock()
        self._analysis_exception: Optional[str] = None
        self._camera_items_cache: list[tuple[str, int]] = [("Camera 0", 0), ("Camera 1", 1), ("Camera 2", 2), ("Camera 3", 3)]
        self._microphone_items_cache: list[tuple[str, Optional[int]]] = [("Default microphone", None)]
        self._subtitle_timeout_ms = 2600
        self._default_subtitle_color = "#F8FAFC"
        self._current_subtitle_color = self._default_subtitle_color
        self._subtitle_font_size_px = 28
        self._current_final_emotion = "NEUTRAL"
        self._current_face_emotion = "NEUTRAL"
        self._current_voice_emotion = "NEUTRAL"
        self._loading_stage = ""
        self._loading_frame = 0

        self._build_ui()
        self._apply_styles()
        self._connect_signals()

        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(self._camera_timer_interval_ms)
        self._capture_timer.timeout.connect(self._grab_frame)

        self._subtitle_hide_timer = QTimer(self)
        self._subtitle_hide_timer.setSingleShot(True)
        self._subtitle_hide_timer.setInterval(self._subtitle_timeout_ms)
        self._subtitle_hide_timer.timeout.connect(self._hide_subtitle_panel)

        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(180)
        self._loading_timer.timeout.connect(self._tick_loading_animation)

        self.set_emotion_state("NEUTRAL", 0.0, 0.0)
        self.set_subtitle("Ready")
        self._apply_layout()
        QTimer.singleShot(0, self._begin_runtime)

    def _build_ui(self) -> None:
        self.central = QWidget(self)
        self.central.setObjectName("CentralRoot")
        self.central.setMouseTracking(True)
        self.setCentralWidget(self.central)

        self.title_bar = GlassPanel(self.central)
        self.title_bar.setObjectName("TitleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(14, 6, 12, 6)
        title_layout.setSpacing(8)

        self.app_title = QLabel("Emotion AI Live Analyzer")
        self.app_title.setObjectName("TitleLabel")
        self.status_chip = QLabel("Ready")
        self.status_chip.setObjectName("StatusChip")

        title_layout.addWidget(self.app_title)
        title_layout.addStretch(1)
        title_layout.addWidget(self.status_chip)

        self.min_button = WindowButton("–")
        self.max_button = WindowButton("⛶")
        self.close_button = WindowButton("×")
        self.min_button.setObjectName("MinButton")
        self.max_button.setObjectName("MaxButton")
        self.close_button.setObjectName("CloseButton")
        self.min_button.clicked.connect(self.showMinimized)
        self.max_button.clicked.connect(self.toggle_fullscreen)
        self.close_button.clicked.connect(self.close)
        title_layout.addWidget(self.min_button)
        title_layout.addWidget(self.max_button)
        title_layout.addWidget(self.close_button)

        self.background = QLabel(self.central)
        self.background.setObjectName("VideoBackground")
        self.background.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.background.setMouseTracking(True)

        self.subtitle = QLabel("Ready", self.central)
        self.subtitle.setObjectName("SubtitleLabel")
        self.subtitle.setWordWrap(True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_panel = GlassPanel(self.central)
        self.subtitle_panel.setObjectName("SubtitlePanel")
        subtitle_layout = QVBoxLayout(self.subtitle_panel)
        subtitle_layout.setContentsMargins(18, 12, 18, 12)
        subtitle_layout.addWidget(self.subtitle)
        subtitle_shadow = QGraphicsDropShadowEffect(self.subtitle_panel)
        subtitle_shadow.setBlurRadius(16)
        subtitle_shadow.setOffset(0, 3)
        subtitle_shadow.setColor(QColor(0, 0, 0, 180))
        self.subtitle_panel.setGraphicsEffect(subtitle_shadow)

        self.video_emotion_panel = GlassPanel(self.central)
        self.video_emotion_panel.setObjectName("VideoEmotionPanel")
        panel_layout = QVBoxLayout(self.video_emotion_panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(10)
        self.video_emotion_title = QLabel("Emotion Summary", self.video_emotion_panel)
        self.video_emotion_title.setObjectName("VideoEmotionTitle")
        self.video_final_emotion_label = QLabel("Final: NEUTRAL", self.video_emotion_panel)
        self.video_face_emotion_label = QLabel("Face: NEUTRAL", self.video_emotion_panel)
        self.video_voice_emotion_label = QLabel("Voice: NEUTRAL", self.video_emotion_panel)
        self.video_final_emotion_label.setObjectName("VideoEmotionValue")
        self.video_face_emotion_label.setObjectName("VideoEmotionValue")
        self.video_voice_emotion_label.setObjectName("VideoEmotionValue")
        panel_layout.addWidget(self.video_emotion_title)
        panel_layout.addWidget(self.video_final_emotion_label)
        panel_layout.addWidget(self.video_face_emotion_label)
        panel_layout.addWidget(self.video_voice_emotion_label)
        self.video_emotion_panel.setVisible(False)

        self.bottom_controls = BottomControls(
            self.central,
            camera_enabled=self.state.camera_enabled,
            mic_enabled=self.state.mic_enabled,
        )
        self.bottom_controls.camera_button.clicked.connect(self.toggle_camera)
        self.bottom_controls.video_button.clicked.connect(self.load_video)
        self.bottom_controls.settings_button.clicked.connect(self.open_settings)

        self.player_controls = PlayerControls(self.central)
        self.player_controls.play_pause_button.clicked.connect(self.toggle_video_play_pause)
        self.player_controls.restart_button.clicked.connect(self.restart_video)
        self.player_controls.setVisible(False)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#CentralRoot {
                background-color: #05070b;
                border: 1px solid rgba(255, 255, 255, 38);
            }
            QFrame#TitleBar {
                background-color: rgba(10, 13, 18, 168);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 15px;
            }
            QLabel#VideoBackground {
                background-color: #05070b;
            }
            QFrame#GlassPanel, QFrame#BottomControls {
                background-color: rgba(12, 16, 23, 142);
                border: 1px solid rgba(255, 255, 255, 26);
                border-radius: 18px;
            }
            QFrame#PlayerControls {
                background-color: rgba(10, 12, 18, 150);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 18px;
            }
            QFrame#SubtitlePanel {
                background-color: rgba(10, 12, 18, 150);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 18px;
            }
            QLabel#SubtitleLabel {
                color: #F8FAFC;
                font-size: 28px;
                font-weight: 700;
                background: transparent;
            }
            QFrame#VideoEmotionPanel {
                background-color: rgba(10, 12, 18, 146);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 14px;
            }
            QLabel#VideoEmotionTitle {
                color: rgba(255, 255, 255, 220);
                font-size: 18px;
                font-weight: 700;
                padding-bottom: 4px;
            }
            QLabel#VideoEmotionValue {
                color: rgba(255, 255, 255, 208);
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#TitleLabel {
                color: rgba(255, 255, 255, 230);
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#StatusChip {
                color: rgba(255, 255, 255, 205);
                background-color: rgba(255, 255, 255, 14);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 10px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton {
                border: none;
            }
            QPushButton#ControlButton {
                border: none;
                border-radius: 29px;
                background-color: rgba(255, 255, 255, 18);
                color: rgba(255, 255, 255, 242);
                font-size: 24px;
                font-weight: 700;
            }
            QPushButton#ControlButton:hover {
                background-color: rgba(255, 255, 255, 30);
            }
            QPushButton#ControlButton[active="true"] {
                background-color: rgba(96, 165, 250, 58);
            }
            QPushButton#MinButton, QPushButton#MaxButton, QPushButton#CloseButton {
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 18px;
                background-color: rgba(255, 255, 255, 12);
                color: rgba(255, 255, 255, 230);
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton#MinButton:hover, QPushButton#MaxButton:hover {
                background-color: rgba(255, 255, 255, 20);
            }
            QPushButton#CloseButton:hover {
                background-color: rgba(239, 68, 68, 92);
                border-color: rgba(239, 68, 68, 120);
            }
            QDialog#SettingsDialog {
                background-color: rgba(9, 12, 17, 245);
                color: #F8FAFC;
            }
            QLabel#SettingsHeader {
                color: rgba(255, 255, 255, 220);
                font-size: 18px;
                font-weight: 700;
                padding-bottom: 4px;
            }
            QComboBox, QCheckBox {
                color: #F8FAFC;
                font-size: 13px;
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 12px;
                padding: 7px 10px;
                min-height: 28px;
            }
            QComboBox::drop-down {
                border: none;
                width: 26px;
            }
            QComboBox QAbstractItemView {
                background-color: #10151d;
                color: #F8FAFC;
                selection-background-color: rgba(96, 165, 250, 120);
                border: 1px solid rgba(255, 255, 255, 18);
            }
            QCheckBox {
                spacing: 10px;
                padding: 4px 2px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid rgba(255, 255, 255, 28);
                background: rgba(255, 255, 255, 8);
            }
            QCheckBox::indicator:checked {
                background: rgba(96, 165, 250, 210);
                border-color: rgba(96, 165, 250, 240);
            }
            QDialogButtonBox QPushButton {
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 14px;
                background-color: rgba(255, 255, 255, 12);
                color: rgba(255, 255, 255, 230);
                min-width: 86px;
                min-height: 34px;
                font-size: 13px;
                font-weight: 700;
                padding: 6px 14px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: rgba(255, 255, 255, 20);
            }
            QDialogButtonBox QPushButton:pressed {
                background-color: rgba(255, 255, 255, 28);
            }
            """
        )

        for button in (
            self.bottom_controls.camera_button,
            self.bottom_controls.mic_button,
            self.bottom_controls.video_button,
            self.bottom_controls.settings_button,
            self.player_controls.play_pause_button,
            self.player_controls.restart_button,
        ):
            button.setObjectName("ControlButton")

    def _connect_signals(self) -> None:
        self.voice_partial.connect(self._on_voice_partial)
        self.voice_final.connect(self._on_voice_final)
        self.voice_error.connect(self._on_voice_error)

    def _begin_runtime(self) -> None:
        if self.state.camera_enabled:
            self._open_camera(self.state.camera_index)
            if self._camera is not None and self._camera.isOpened():
                self._capture_timer.start()
            else:
                self._show_placeholder_frame()
        else:
            self._show_placeholder_frame()
            if not self.state.mic_enabled:
                self.status_chip.setText("Inputs off")

        if self.state.mic_enabled:
            self._start_voice_recognition()

    def _apply_layout(self) -> None:
        self.background.setGeometry(self.rect())
        self._position_title_bar()
        self._position_subtitle()
        self._position_video_emotion_panel()
        self._position_player_controls()
        self._position_bottom_panel()
        self.title_bar.raise_()
        self.video_emotion_panel.raise_()
        self.subtitle_panel.raise_()
        self.subtitle.raise_()
        self.player_controls.raise_()
        self.bottom_controls.raise_()

    def _position_title_bar(self) -> None:
        width = self.width() - 32
        self.title_bar.setGeometry(16, 12, width, 42)

    def _position_subtitle(self) -> None:
        text = self.state.subtitle.strip()
        is_video_mode = self._source_mode == "video"
        max_width = min(760, int(self.width() * 0.56)) if is_video_mode else min(940, int(self.width() * 0.68))
        min_width = 300 if is_video_mode else 360
        self._subtitle_font_size_px = 22 if is_video_mode else 28
        self.subtitle.setStyleSheet(
            f"color: {self._current_subtitle_color}; font-size: {self._subtitle_font_size_px}px;"
        )
        if text:
            metrics = self.subtitle.fontMetrics()
            bounds = metrics.boundingRect(0, 0, max_width - 32, 800, int(Qt.TextFlag.TextWordWrap), text)
            extra_w = 52 if is_video_mode else 64
            extra_h = 30 if is_video_mode else 36
            min_h = 70 if is_video_mode else 84
            max_h = 190 if is_video_mode else 228
            panel_width = max(min_width, min(max_width, bounds.width() + extra_w))
            panel_height = max(min_h, min(max_h, bounds.height() + extra_h))
        else:
            panel_width = min_width
            panel_height = 70 if is_video_mode else 84

        x = int((self.width() - panel_width) / 2)
        controls_gap = 18 if is_video_mode else (92 if self.player_controls.isVisible() else 28)
        y = self.height() - panel_height - controls_gap
        self.subtitle_panel.setGeometry(x, y, panel_width, panel_height)
        self.subtitle.setGeometry(20, 12, panel_width - 40, panel_height - 24)

    def _position_video_emotion_panel(self) -> None:
        width = 300
        height = 176
        x = 18
        y = self.title_bar.geometry().bottom() + 12
        self.video_emotion_panel.setGeometry(x, y, width, height)

    def _position_player_controls(self) -> None:
        width = self.player_controls.width()
        height = self.player_controls.height()
        if self._source_mode == "video":
            x = 20
            y = self.height() - height - 20
        else:
            x = int((self.width() - width) / 2)
            y = self.subtitle_panel.geometry().bottom() + 10
        self.player_controls.setGeometry(x, y, width, height)

    def _bottom_geometry(self) -> QRect:
        width = self.bottom_controls.width()
        height = self.bottom_controls.height()
        x = self.width() - width - 24
        y = int((self.height() - height) / 2)
        return QRect(x, y, width, height)

    def _position_bottom_panel(self) -> None:
        geometry = self._bottom_geometry()
        self.bottom_controls.setGeometry(geometry)

    def _grab_frame(self) -> None:
        if self._source_mode == "camera" and not self.state.camera_enabled:
            return
        if self._source_mode == "video" and self._video_paused:
            return
        if self._camera is None or not self._camera.isOpened():
            self._show_placeholder_frame()
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            if self._source_mode == "video":
                self._video_ended = True
                self._video_paused = True
                self.player_controls.play_pause_button.setText("▶")
                self.status_chip.setText("Video ended")
                if self._last_frame is not None:
                    self.update_frame(self._last_frame)
                return
            self._show_placeholder_frame()
            return
        if self._source_mode == "video":
            analyzed = self._render_preprocessed_video_frame(frame)
        else:
            analyzed = self._analyze_frame(frame)
        self._current_frame = analyzed
        self._last_frame = analyzed
        self.update_frame(analyzed)

    def _render_preprocessed_video_frame(self, frame: np.ndarray) -> np.ndarray:
        pipeline = self._pipeline
        if pipeline is None:
            return frame

        frame_idx = self._video_frame_index
        self._video_frame_index += 1

        results = self._video_face_results_by_frame.get(frame_idx)
        if results is not None:
            self._last_video_results = results
        else:
            results = self._last_video_results

        output = pipeline.draw_results(frame, results)

        if results:
            best = max(results, key=lambda item: float(item.get("confidence", 0.0)))
            face_emotion = str(best.get("emotion", "neutral")).upper()
            face_confidence = float(best.get("confidence", 0.0))
        else:
            face_emotion = "NO FACE"
            face_confidence = 0.0

        sec = frame_idx / max(self._video_fps, 1e-6)
        cue = self._subtitle_cue_for_time(sec)

        if cue is not None:
            self.set_subtitle(cue.text, color=cue.color, force_visible=True)
            voice_confidence = cue.confidence
            text_emotion = cue.text_emotion
        else:
            self.set_subtitle("", force_visible=False)
            voice_confidence = 0.0
            text_emotion = "NEUTRAL"

        final_emotion, _ = self._summarize_emotion(
            face_emotion,
            face_confidence,
            text_emotion,
            voice_confidence,
        )
        self.set_emotion_state(final_emotion, face_confidence, voice_confidence)
        self._update_video_emotion_panel(final_emotion, face_emotion, text_emotion)
        self.status_chip.setText("Video analysis running")
        return output

    def _update_video_emotion_panel(self, final_emotion: str, face_emotion: str, voice_emotion: str) -> None:
        self._current_final_emotion = (final_emotion or "NEUTRAL").upper()
        self._current_face_emotion = (face_emotion or "NEUTRAL").upper()
        self._current_voice_emotion = (voice_emotion or "NEUTRAL").upper()
        self.video_final_emotion_label.setText(f"Final: {self._current_final_emotion}")
        self.video_face_emotion_label.setText(f"Face: {self._current_face_emotion}")
        self.video_voice_emotion_label.setText(f"Voice: {self._current_voice_emotion}")
        self.video_emotion_panel.setVisible(
            self._source_mode == "video" and self.state.show_video_emotion_panel
        )

    def _subtitle_cue_for_time(self, sec: float) -> Optional[SubtitleCue]:
        if not self._video_subtitle_cues:
            self._active_video_cue_index = -1
            return None

        start_index = max(self._active_video_cue_index, 0)
        for i in range(start_index, len(self._video_subtitle_cues)):
            cue = self._video_subtitle_cues[i]
            if cue.start <= sec <= cue.end:
                self._active_video_cue_index = i
                return cue
            if sec < cue.start:
                break

        if self._active_video_cue_index >= 0:
            cue = self._video_subtitle_cues[self._active_video_cue_index]
            if cue.start <= sec <= cue.end:
                return cue

        self._active_video_cue_index = -1
        return None

    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            pipeline = self._ensure_pipeline()
        except Exception:
            pipeline = None
        if pipeline is None:
            return frame

        results = pipeline.process_frame(frame)

        output = pipeline.draw_results(frame, results)
        if results:
            best = max(results, key=lambda item: float(item.get("confidence", 0.0)))
            emotion = str(best.get("emotion", "neutral")).upper()
            confidence = float(best.get("confidence", 0.0))
        else:
            emotion = "NO FACE"
            confidence = 0.0

        voice_confidence = 0.0
        if self.state.mic_enabled:
            voice_confidence = max(self.state.voice_confidence, self._live_text_confidence)

        final_emotion, _ = self._summarize_emotion(
            emotion,
            confidence,
            self._live_text_emotion,
            voice_confidence,
        )
        self.set_emotion_state(final_emotion, confidence, voice_confidence)
        if results:
            self.status_chip.setText("Live analysis running")
        else:
            self.status_chip.setText("No face")
        return output

    def _summarize_emotion(
        self,
        face_emotion: str,
        face_confidence: float,
        voice_emotion: str,
        voice_confidence: float,
    ) -> tuple[str, float]:
        face = self._face_to_polarity(face_emotion)
        voice = (voice_emotion or "NEUTRAL").upper()
        face_score = max(0.0, min(1.0, float(face_confidence)))
        voice_score = max(0.0, min(1.0, float(voice_confidence)))

        # Text/voice has higher influence than face in the final decision.
        face_weight = 0.35
        text_weight = 0.65

        face_value = self._emotion_to_signed(face) * face_score * face_weight
        voice_value = self._emotion_to_signed(voice) * voice_score * text_weight
        combined = face_value + voice_value

        if abs(combined) < 0.12:
            return "NEUTRAL", min(1.0, max(face_score, voice_score))
        if combined > 0:
            return "POSITIVE", min(1.0, abs(combined))
        return "NEGATIVE", min(1.0, abs(combined))

    def _face_to_polarity(self, emotion: str) -> str:
        value = (emotion or "NEUTRAL").upper()
        if value in {"HAPPY", "SURPRISE"}:
            return "POSITIVE"
        if value in {"ANGER", "ANGRY", "DISGUST", "FEAR", "SAD"}:
            return "NEGATIVE"
        return "NEUTRAL"

    def _emotion_to_signed(self, emotion: str) -> float:
        value = (emotion or "NEUTRAL").upper()
        if value == "POSITIVE":
            return 1.0
        if value == "NEGATIVE":
            return -1.0
        return 0.0

    def _run_async_inference(self, pipeline: EmotionPipeline, frame: np.ndarray) -> None:
        try:
            results = pipeline.process_frame(frame)
            with self._analysis_lock:
                self._cached_results = results
        except Exception as exc:
            self._analysis_exception = str(exc)
        finally:
            self._analysis_busy = False

    def _ensure_pipeline(self) -> Optional[EmotionPipeline]:
        if self._pipeline is not None:
            return self._pipeline
        if self._analysis_loading:
            return None
        self._analysis_loading = True
        self.status_chip.setText("Loading model")
        model_path = self._default_model_path()
        try:
            self._pipeline = EmotionPipeline(
                model_checkpoint=model_path,
                face_confidence=0.85,
                input_size=112,
            )
            self.status_chip.setText("Model ready")
            return self._pipeline
        except Exception as exc:
            self.status_chip.setText("Analysis failed")
            self.set_subtitle(f"Model load failed: {exc}")
            return None
        finally:
            self._analysis_loading = False

    def _show_placeholder_frame(self) -> None:
        width, height = 1280, 720
        background = np.zeros((height, width, 3), dtype=np.uint8)
        gradient = np.linspace(12, 44, width, dtype=np.uint8)
        background[:, :, 0] = gradient
        background[:, :, 1] = gradient // 2
        background[:, :, 2] = gradient // 3
        self.update_frame(background)

    def update_frame(self, cv_img: object) -> None:
        if not isinstance(cv_img, np.ndarray):
            return
        if cv_img.ndim == 2:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_img.shape
        image = QImage(rgb_img.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        self._current_frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if cv_img.ndim == 3 else rgb_img
        self.background.setPixmap(
            pixmap.scaled(
                self.background.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.background.setGeometry(self.rect())

    def set_emotion_state(self, emotion: str, face_confidence: float, voice_confidence: float) -> None:
        self.state.emotion = emotion.upper().strip() or "NEUTRAL"
        self.state.face_confidence = max(0.0, min(1.0, float(face_confidence)))
        self.state.voice_confidence = max(0.0, min(1.0, float(voice_confidence)))

    def set_subtitle(self, text: str, color: Optional[str] = None, force_visible: bool = False) -> None:
        clean = text.strip() if text else ""
        self.state.subtitle = clean
        self.subtitle.setText(clean)
        subtitle_color = color or self._default_subtitle_color
        self._current_subtitle_color = subtitle_color
        self.subtitle.setStyleSheet(
            f"color: {subtitle_color}; font-size: {self._subtitle_font_size_px}px;"
        )
        visible = bool(clean) and self.state.show_subtitles and (
            self.state.mic_enabled or force_visible or self._source_mode == "video"
        )
        self.subtitle_panel.setVisible(visible)
        self._position_subtitle()
        if visible and not force_visible and self._source_mode != "video":
            self._subtitle_hide_timer.start()
        else:
            self._subtitle_hide_timer.stop()
        if clean and self.state.show_subtitles and (self.state.mic_enabled or self._source_mode == "video"):
            self.status_chip.setText("Mic on")

    def _hide_subtitle_panel(self) -> None:
        self.subtitle_panel.setVisible(False)

    def toggle_camera(self) -> None:
        if self._source_mode == "video":
            self._source_mode = "camera"
            self._video_path = None
            self._video_paused = False
            self._video_ended = False
            self._analysis_busy = False
            self._cached_results = []
            self._video_frame_index = 0
            self.player_controls.setVisible(False)
            self.video_emotion_panel.setVisible(False)
            self.state.camera_enabled = True
            self.bottom_controls.camera_button.setChecked(True)
            self.bottom_controls._refresh_button_style()
            self._open_camera(self.state.camera_index)
            self._capture_timer.setInterval(self._camera_timer_interval_ms)
            if self._camera is not None and self._camera.isOpened():
                self._capture_timer.start()
            else:
                self._show_placeholder_frame()
            if self.state.mic_enabled and self._voice_recognizer is None:
                self._start_voice_recognition()
            self._apply_layout()
            return

        self.state.camera_enabled = not self.state.camera_enabled
        self.bottom_controls.camera_button.setChecked(self.state.camera_enabled)
        self.bottom_controls._refresh_button_style()
        if self.state.camera_enabled:
            self._source_mode = "camera"
            self._video_path = None
            self._video_paused = False
            self._video_ended = False
            self._analysis_busy = False
            self._cached_results = []
            self._video_frame_index = 0
            self.player_controls.setVisible(False)
            self.video_emotion_panel.setVisible(False)
            self._open_camera(self.state.camera_index)
            self._capture_timer.setInterval(self._camera_timer_interval_ms)
            if self._camera is not None and self._camera.isOpened():
                self._capture_timer.start()
            else:
                self._show_placeholder_frame()
        else:
            self._capture_timer.stop()
            self._release_camera()
            self._show_placeholder_frame()
        self._apply_layout()

    def toggle_mic(self) -> None:
        self.state.mic_enabled = not self.state.mic_enabled
        self.bottom_controls.mic_button.setChecked(self.state.mic_enabled)
        self.bottom_controls._refresh_button_style()
        if self.state.mic_enabled:
            self._start_voice_recognition()
        else:
            self._stop_voice_recognition()
            self._live_text_emotion = "NEUTRAL"
            self._live_text_confidence = 0.0
            self._subtitle_hide_timer.stop()
            self.subtitle_panel.setVisible(False)
        self.status_chip.setText("Mic on" if self.state.mic_enabled else "Mic off")

    def load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if not path:
            return

        self._disable_live_inputs_for_video_mode()

        self._start_loading_animation("Preprocessing video")
        self.set_subtitle("Preparing predefined face/text emotions", color="#FDE68A", force_visible=True)
        QApplication.processEvents()

        try:
            self._preprocess_video(path)
        except Exception as exc:
            self._stop_loading_animation()
            self.status_chip.setText("Preprocess failed")
            self.set_subtitle(str(exc), color="#FCA5A5", force_visible=True)
            return
        self._stop_loading_animation()

        if not self._open_video(path):
            self.status_chip.setText("Video open failed")
            return

        self._source_mode = "video"
        self._video_path = path
        self._video_paused = False
        self._video_ended = False
        self._analysis_busy = False
        self._cached_results = []
        self._last_video_results = []
        self._active_video_cue_index = -1
        self._video_frame_index = 0
        self.player_controls.play_pause_button.setText("⏸")
        self.player_controls.setVisible(True)
        self.bottom_controls.camera_button.setChecked(False)
        self.bottom_controls.mic_button.setChecked(False)
        self.bottom_controls._refresh_button_style()
        self._capture_timer.setInterval(self._video_timer_interval_ms)
        self._capture_timer.start()
        self.status_chip.setText(f"Video ready: {Path(path).name}")
        self.set_subtitle("", force_visible=False)
        self._apply_layout()

    def _start_loading_animation(self, stage: str) -> None:
        self._loading_stage = stage
        self._loading_frame = 0
        if not self._loading_timer.isActive():
            self._loading_timer.start()
        self._tick_loading_animation()

    def _set_loading_stage(self, stage: str) -> None:
        self._loading_stage = stage
        self._loading_frame = 0
        self._tick_loading_animation()

    def _tick_loading_animation(self) -> None:
        spinner = ("◜", "◠", "◝", "◞", "◡", "◟")
        spin = spinner[self._loading_frame % len(spinner)]
        text = f"{self._loading_stage} {spin}".strip()
        if text:
            self.status_chip.setText(text)
        self._loading_frame += 1

    def _stop_loading_animation(self) -> None:
        if self._loading_timer.isActive():
            self._loading_timer.stop()
        self._loading_stage = ""
        self._loading_frame = 0

    def _disable_live_inputs_for_video_mode(self) -> None:
        self.state.camera_enabled = False
        self.state.mic_enabled = False
        self.bottom_controls.camera_button.setChecked(False)
        self.bottom_controls.mic_button.setChecked(False)
        self.bottom_controls._refresh_button_style()
        if self._capture_timer.isActive():
            self._capture_timer.stop()
        self._stop_voice_recognition()
        self._release_camera()
        self._live_text_emotion = "NEUTRAL"
        self._live_text_confidence = 0.0
        self.video_emotion_panel.setVisible(False)

    def _preprocess_video(self, path: str) -> None:
        pipeline = self._ensure_pipeline()
        if pipeline is None:
            raise RuntimeError("Emotion model is not available.")

        if not self._voice_model_path:
            raise RuntimeError("VOSK model not found. Text emotion preprocessing requires speech model.")

        # RuBERT is optional here: if unavailable, we keep subtitles and fallback to neutral text emotion.
        self._ensure_text_emotion_analyzer()

        self._video_face_results_by_frame = {}
        self._video_subtitle_cues = []
        self._active_video_cue_index = -1

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        try:
            self._set_loading_stage("Extracting audio")
            QApplication.processEvents()
            extract_audio_track(path, wav_path)

            self._set_loading_stage("Transcribing audio")
            QApplication.processEvents()
            transcriber = VoskAudioFileRecognizer(self._voice_model_path)
            segments = transcriber.transcribe_wav(wav_path)
            self._video_subtitle_cues = self._build_subtitle_cues(segments)
            if self._text_emotion_analyzer is None and self._rubert_init_error:
                self.set_subtitle(
                    f"RuBERT unavailable, neutral subtitles: {self._rubert_init_error}",
                    color="#FDE68A",
                    force_visible=True,
                )

            self._set_loading_stage("Analyzing face emotions")
            QApplication.processEvents()
            self._precompute_face_timeline(path, pipeline)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _build_subtitle_cues(self, segments: list[TranscriptSegment]) -> list[SubtitleCue]:
        analyzer = self._ensure_text_emotion_analyzer()

        cues: list[SubtitleCue] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if analyzer is None:
                info = {
                    "emotion": "NEUTRAL",
                    "confidence": 0.0,
                    "color": "#E2E8F0",
                }
            else:
                info = analyzer.analyze(text)
            cues.append(
                SubtitleCue(
                    start=max(0.0, float(segment.start)),
                    end=max(float(segment.end), float(segment.start) + 0.12),
                    text=text,
                    text_emotion=str(info["emotion"]),
                    confidence=float(info["confidence"]),
                    color=str(info["color"]),
                )
            )
        return cues

    def _precompute_face_timeline(self, path: str, pipeline: EmotionPipeline) -> None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError("Unable to open video for preprocessing.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0
        self._video_fps = fps

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._video_total_frames = max(0, total)

        frame_idx = 0
        sampled: dict[int, list[dict]] = {}
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if frame_idx % self._video_analyze_every_n == 0:
                sampled[frame_idx] = pipeline.process_frame(frame)

            frame_idx += 1
            if frame_idx % 50 == 0:
                if total > 0:
                    progress = int((frame_idx / total) * 100)
                    self._set_loading_stage(f"Face analysis {progress}%")
                else:
                    self._set_loading_stage(f"Face analysis frame {frame_idx}")
                QApplication.processEvents()

        capture.release()
        self._video_face_results_by_frame = sampled

    def toggle_video_play_pause(self) -> None:
        if self._source_mode != "video":
            return
        if self._video_ended:
            self.status_chip.setText("Press restart")
            return
        self._video_paused = not self._video_paused
        if self._video_paused:
            self.player_controls.play_pause_button.setText("▶")
            self.status_chip.setText("Paused")
        else:
            self.player_controls.play_pause_button.setText("⏸")
            self.status_chip.setText("Playing")

    def restart_video(self) -> None:
        if self._source_mode != "video":
            return
        if self._camera is None or not self._camera.isOpened():
            if not self._video_path or not self._open_video(self._video_path):
                self.status_chip.setText("Video reopen failed")
                return
        self._camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._video_paused = False
        self._video_ended = False
        self._analysis_busy = False
        self._cached_results = []
        self._last_video_results = []
        self._active_video_cue_index = -1
        self._video_frame_index = 0
        self.player_controls.play_pause_button.setText("⏸")
        self._capture_timer.start()
        self.status_chip.setText("Restarted")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.max_button.setText("⛶")
        else:
            self.showFullScreen()
            self.max_button.setText("🗗")

    def open_settings(self) -> None:
        camera_items = self._available_camera_items()
        microphone_items = self._available_microphone_items()
        dialog = SettingsDialog(
            camera_items=camera_items,
            microphone_items=microphone_items,
            current_camera=self.state.camera_index,
            current_microphone=self.state.mic_device,
            show_subtitles=self.state.show_subtitles,
            show_video_emotion_panel=self.state.show_video_emotion_panel,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_camera = dialog.selected_camera()
        new_microphone = dialog.selected_microphone()
        self.state.show_subtitles = dialog.subtitles_enabled()
        self.state.show_video_emotion_panel = dialog.video_emotion_panel_enabled()
        camera_changed = new_camera != self.state.camera_index
        mic_changed = new_microphone != self.state.mic_device

        self.state.camera_index = new_camera
        self.state.mic_device = new_microphone

        if camera_changed:
            self._restart_camera(new_camera)

        if mic_changed and self.state.mic_enabled:
            self._stop_voice_recognition()
            self._start_voice_recognition()

        if not self.state.show_subtitles or not self.state.mic_enabled:
            self._subtitle_hide_timer.stop()
            self.subtitle_panel.setVisible(False)
        elif self.state.subtitle:
            self.subtitle_panel.setVisible(True)

        self.video_emotion_panel.setVisible(
            self._source_mode == "video" and self.state.show_video_emotion_panel
        )

        self.status_chip.setText("Settings updated")
        self.set_subtitle("Devices updated")

    def _default_model_path(self) -> Optional[str]:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "checkpoints_production" / "best_emotion_model.pt",
            root / "checkpoints" / "best_emotion_model.pt",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _default_vosk_model_path(self) -> Optional[str]:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "voskmodel" / "vosk-model-small-ru-0.22",
            root / "vosk-model-small-ru-0.22",
        ]
        for path in candidates:
            if self._is_valid_vosk_model_dir(path):
                return str(path)
        return None

    def _default_rubert_model_path(self) -> Optional[str]:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "rubert-base-cased-russian-sentiment",
            root / "models" / "rubert-base-cased-russian-sentiment",
            root / "models" / "rubert",
            root.parent / "rubert-base-cased-russian-sentiment",
        ]

        for path in candidates:
            if self._is_valid_rubert_model_dir(path):
                return str(path)

        # Fallback scan: try to find RuBERT-like folders in project root and models/.
        scan_roots = [root, root / "models"]
        for scan_root in scan_roots:
            if not scan_root.exists() or not scan_root.is_dir():
                continue
            for child in scan_root.iterdir():
                if not child.is_dir():
                    continue
                name = child.name.lower()
                if "rubert" in name or "sentiment" in name:
                    if self._is_valid_rubert_model_dir(child):
                        return str(child)
        return None

    def _ensure_text_emotion_analyzer(self) -> Optional[RuBertTextEmotionAnalyzer]:
        if self._text_emotion_analyzer is not None:
            return self._text_emotion_analyzer
        if not self._rubert_model_path:
            self._rubert_init_error = "model directory not found"
            return None
        try:
            self._text_emotion_analyzer = RuBertTextEmotionAnalyzer(self._rubert_model_path)
            self._rubert_init_error = None
            return self._text_emotion_analyzer
        except Exception as exc:
            self._rubert_init_error = str(exc)
            self.status_chip.setText("Text model failed")
            self.set_subtitle(f"Text model load failed: {exc}", color="#FCA5A5", force_visible=True)
            return None

    def _is_valid_vosk_model_dir(self, path: Path) -> bool:
        required = ["am", "conf", "graph"]
        return path.exists() and path.is_dir() and all((path / item).exists() for item in required)

    def _is_valid_rubert_model_dir(self, path: Path) -> bool:
        required = ["config.json", "tokenizer.json", "vocab.txt"]
        has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        return path.exists() and path.is_dir() and all((path / item).exists() for item in required) and has_weights

    def _available_camera_items(self) -> list[tuple[str, int]]:
        items = list(self._camera_items_cache)
        if self.state.camera_index not in [value for _, value in items]:
            items.insert(0, (f"Camera {self.state.camera_index}", self.state.camera_index))
        return items

    def _available_microphone_items(self) -> list[tuple[str, Optional[int]]]:
        items: list[tuple[str, Optional[int]]] = [("Default microphone", None)]
        if sd is None:
            return items
        try:
            devices = sd.query_devices()
        except Exception:
            return items
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) > 0:
                name = str(device.get("name", f"Device {index}"))
                items.append((f"{name} ({index})", index))
        return items

    def _create_capture(self, index: int) -> cv2.VideoCapture:
        if sys.platform.startswith("win"):
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(index)

    def _open_camera(self, index: int) -> bool:
        self._release_camera()
        capture = self._create_capture(index)
        if not capture.isOpened():
            self._camera = None
            self.status_chip.setText(f"Camera {index} unavailable")
            return False
        self._camera = capture
        self.state.camera_index = index
        self.status_chip.setText(f"Camera {index}")
        return True

    def _open_video(self, path: str) -> bool:
        self._release_camera()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self._camera = None
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0
        self._video_fps = fps
        self._video_timer_interval_ms = max(12, min(42, int(1000.0 / fps)))
        self._camera = capture
        return True

    def _restart_camera(self, index: int) -> None:
        was_running = self._capture_timer.isActive()
        self._capture_timer.stop()
        if self._open_camera(index) and self.state.camera_enabled and was_running:
            self._capture_timer.start()
        elif self.state.camera_enabled:
            self._show_placeholder_frame()

    def _release_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _start_voice_recognition(self) -> None:
        if self._voice_recognizer is not None or not self._voice_model_path:
            return
        try:
            self._voice_recognizer = VoskMicrophoneRecognizer(
                model_path=self._voice_model_path,
                device=self.state.mic_device,
            )
            self._voice_recognizer.start(
                on_partial=lambda text: self.voice_partial.emit(text),
                on_final=lambda text: self.voice_final.emit(text),
                on_error=lambda text: self.voice_error.emit(text),
            )
            self.status_chip.setText("Mic on")
        except Exception as exc:
            self._voice_recognizer = None
            self.state.mic_enabled = False
            self.bottom_controls.mic_button.setChecked(False)
            self.bottom_controls._refresh_button_style()
            self.status_chip.setText("Mic error")
            self.set_subtitle(f"Voice disabled: {exc}")

    def _stop_voice_recognition(self) -> None:
        if self._voice_recognizer is None:
            return
        try:
            self._voice_recognizer.stop()
        finally:
            self._voice_recognizer = None

    def _on_voice_partial(self, text: str) -> None:
        clean = text.strip()
        if clean:
            if self._source_mode == "camera":
                return
            self.set_subtitle(clean, color="#F8FAFC")

    def _on_voice_final(self, text: str) -> None:
        clean = text.strip()
        if clean:
            analyzer = self._ensure_text_emotion_analyzer()
            if analyzer is not None:
                info = analyzer.analyze(clean)
                self._live_text_emotion = str(info.get("emotion", "NEUTRAL"))
                self._live_text_confidence = float(info.get("confidence", 0.0))
            if self._source_mode == "camera":
                return
            self.set_subtitle(clean, color="#F8FAFC")

    def _on_voice_error(self, text: str) -> None:
        self._live_text_emotion = "NEUTRAL"
        self._live_text_confidence = 0.0
        self.status_chip.setText("Mic error")
        if self._source_mode == "camera":
            return
        self.set_subtitle(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_layout()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._capture_timer.isActive():
            self._capture_timer.stop()
        self._release_camera()
        self._stop_voice_recognition()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Emotion AI Live Analyzer")
    app.setStyle("Fusion")
    window = EmotionAILiveAnalyzer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())