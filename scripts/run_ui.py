#!/usr/bin/env python3
from __future__ import annotations

import sys
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

from src.audio.speech_to_text import VoskMicrophoneRecognizer
from src.pipeline import EmotionPipeline


EMOTION_COLORS = {
    "NEUTRAL": "#E5E7EB",
    "NO FACE": "#AEB7C4",
    "HAPPY": "#86EFAC",
    "SAD": "#93C5FD",
    "ANGRY": "#F87171",
    "SURPRISE": "#FCD34D",
    "FEAR": "#C4B5FD",
    "DISGUST": "#A3E635",
}


@dataclass
class AnalyzerState:
    emotion: str = "NEUTRAL"
    face_confidence: float = 0.0
    voice_confidence: float = 0.0
    subtitle: str = "Ready"
    camera_enabled: bool = True
    mic_enabled: bool = True
    camera_index: int = 0
    mic_device: Optional[int] = None
    show_subtitles: bool = True


class GlassPanel(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(42)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)


class ConfidenceBar(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._accent = QColor("#60A5FA")
        self.setFixedHeight(44)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricLabel")
        self.value_label = QLabel("0%")
        self.value_label.setObjectName("MetricValue")
        row.addWidget(self.title_label)
        row.addStretch(1)
        row.addWidget(self.value_label)
        layout.addLayout(row)

        self.track = QFrame()
        self.track.setObjectName("ConfidenceTrack")
        self.track.setFixedHeight(6)
        self.fill = QFrame(self.track)
        self.fill.setObjectName("ConfidenceFill")
        self.fill.setGeometry(0, 0, 0, 6)
        layout.addWidget(self.track)

    def setValue(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.value_label.setText(f"{int(self._value * 100)}%")
        width = max(0, int(self.track.width() * self._value))
        self.fill.setGeometry(0, 0, width, self.track.height())
        self.fill.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255,255,255,0.90), stop:1 {self._accent.name()}); border-radius: 3px;"
        )

    def setAccent(self, color: QColor) -> None:
        self._accent = QColor(color)
        self.setValue(self._value)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.setValue(self._value)


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
        self.subtitle_checkbox.setChecked(show_subtitles)

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


class BottomControls(GlassPanel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("BottomControls")
        self.setFixedSize(92, 304)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        self.camera_button = RoundButton("📷")
        self.mic_button = RoundButton("🎤")
        self.video_button = RoundButton("🎬")
        self.settings_button = RoundButton("⚙")
        self.camera_button.setCheckable(True)
        self.mic_button.setCheckable(True)
        self.video_button.setCheckable(False)
        self.settings_button.setCheckable(False)
        self.camera_button.setChecked(True)
        self.mic_button.setChecked(True)

        layout.addStretch(1)
        layout.addWidget(self.camera_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mic_button, alignment=Qt.AlignmentFlag.AlignCenter)
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
        self._voice_model_path = self._default_vosk_model_path()
        self._analysis_loading = False
        self._source_mode = "camera"
        self._video_path: Optional[str] = None
        self._video_paused = False
        self._video_ended = False
        self._last_frame: Optional[np.ndarray] = None
        self._cached_results: list[dict] = []
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

        self.left_panel = GlassPanel(self.central)
        self.left_panel.setObjectName("LeftPanel")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(18)

        self.panel_title = QLabel("FINAL EMOTION")
        self.panel_title.setObjectName("PanelHeader")
        self.final_emotion_label = QLabel("NEUTRAL")
        self.final_emotion_label.setObjectName("FinalEmotion")
        self.final_emotion_label.setWordWrap(True)

        self.face_bar = ConfidenceBar("Face Confidence")
        self.voice_bar = ConfidenceBar("Voice Confidence")

        left_layout.addWidget(self.panel_title)
        left_layout.addWidget(self.final_emotion_label)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.face_bar)
        left_layout.addWidget(self.voice_bar)
        left_layout.addStretch(1)

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

        self.bottom_controls = BottomControls(self.central)
        self.bottom_controls.camera_button.clicked.connect(self.toggle_camera)
        self.bottom_controls.mic_button.clicked.connect(self.toggle_mic)
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
            QFrame#GlassPanel, QFrame#BottomControls, QFrame#LeftPanel {
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
            QLabel#PanelHeader {
                color: rgba(255, 255, 255, 145);
                font-size: 14px;
                letter-spacing: 4px;
                font-weight: 700;
            }
            QLabel#FinalEmotion {
                color: #E5E7EB;
                font-size: 44px;
                font-weight: 800;
                letter-spacing: 2px;
            }
            QLabel#MetricLabel {
                color: rgba(255, 255, 255, 210);
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#MetricValue {
                color: rgba(255, 255, 255, 220);
                font-size: 13px;
                font-weight: 700;
            }
            QFrame#ConfidenceTrack {
                background-color: rgba(255, 255, 255, 28);
                border-radius: 3px;
            }
            QLabel#SubtitleLabel {
                color: #F8FAFC;
                font-size: 28px;
                font-weight: 700;
                background: transparent;
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

        if self.state.mic_enabled:
            self._start_voice_recognition()

    def _apply_layout(self) -> None:
        self.background.setGeometry(self.rect())
        self._position_title_bar()
        self._position_left_panel()
        self._position_subtitle()
        self._position_player_controls()
        self._position_bottom_panel()
        self.title_bar.raise_()
        self.left_panel.raise_()
        self.subtitle_panel.raise_()
        self.subtitle.raise_()
        self.player_controls.raise_()
        self.bottom_controls.raise_()

    def _position_title_bar(self) -> None:
        width = self.width() - 32
        self.title_bar.setGeometry(16, 12, width, 42)

    def _position_left_panel(self) -> None:
        width = min(390, max(300, int(self.width() * 0.28)))
        height = min(284, max(240, int(self.height() * 0.38)))
        top = self.title_bar.geometry().bottom() + 16
        self.left_panel.setGeometry(24, top, width, height)

    def _position_subtitle(self) -> None:
        text = self.state.subtitle.strip()
        max_width = min(760, int(self.width() * 0.54))
        min_width = 260
        if text:
            metrics = self.subtitle.fontMetrics()
            bounds = metrics.boundingRect(0, 0, max_width - 32, 800, int(Qt.TextFlag.TextWordWrap), text)
            panel_width = max(min_width, min(max_width, bounds.width() + 44))
            panel_height = max(64, min(168, bounds.height() + 28))
        else:
            panel_width = min_width
            panel_height = 64

        x = int((self.width() - panel_width) / 2)
        controls_gap = 92 if self.player_controls.isVisible() else 28
        y = self.height() - panel_height - controls_gap
        self.subtitle_panel.setGeometry(x, y, panel_width, panel_height)
        self.subtitle.setGeometry(16, 10, panel_width - 32, panel_height - 20)

    def _position_player_controls(self) -> None:
        width = self.player_controls.width()
        height = self.player_controls.height()
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
        analyzed = self._analyze_frame(frame)
        self._current_frame = analyzed
        self._last_frame = analyzed
        self.update_frame(analyzed)

    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            pipeline = self._ensure_pipeline()
        except Exception:
            pipeline = None
        if pipeline is None:
            return frame

        if self._source_mode == "video":
            self._video_frame_index += 1
            analyze_now = (self._video_frame_index % self._video_analyze_every_n == 0) or not self._cached_results
            if analyze_now and not self._analysis_busy:
                frame_copy = frame.copy()
                self._analysis_busy = True
                threading.Thread(target=self._run_async_inference, args=(pipeline, frame_copy), daemon=True).start()
            with self._analysis_lock:
                results = list(self._cached_results)
            if self._analysis_exception:
                self.status_chip.setText("Analysis warning")
                self._analysis_exception = None
        else:
            results = pipeline.process_frame(frame)

        output = pipeline.draw_results(frame, results)
        if results:
            best = max(results, key=lambda item: float(item.get("confidence", 0.0)))
            emotion = str(best.get("emotion", "neutral")).upper()
            confidence = float(best.get("confidence", 0.0))
        else:
            emotion = "NO FACE"
            confidence = 0.0

        voice_confidence = 0.0 if not self.state.mic_enabled else max(self.state.voice_confidence, 0.15)
        self.set_emotion_state(emotion, confidence, voice_confidence)
        self.status_chip.setText("Analyzing" if results else "No face")
        return output

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

    def _emotion_color(self, emotion: str) -> QColor:
        return QColor(EMOTION_COLORS.get(emotion.upper(), "#E5E7EB"))

    def set_emotion_state(self, emotion: str, face_confidence: float, voice_confidence: float) -> None:
        self.state.emotion = emotion.upper().strip() or "NEUTRAL"
        self.state.face_confidence = max(0.0, min(1.0, float(face_confidence)))
        self.state.voice_confidence = max(0.0, min(1.0, float(voice_confidence)))
        accent = self._emotion_color(self.state.emotion)
        label = "No face" if self.state.emotion == "NO FACE" else self.state.emotion
        self.final_emotion_label.setText(label)
        self.final_emotion_label.setStyleSheet(f"color: {accent.name()};")
        self.face_bar.setAccent(accent)
        self.face_bar.setValue(self.state.face_confidence)
        self.voice_bar.setAccent(QColor(96, 165, 250))
        self.voice_bar.setValue(self.state.voice_confidence)

    def set_subtitle(self, text: str) -> None:
        clean = text.strip() if text else ""
        self.state.subtitle = clean
        self.subtitle.setText(clean)
        visible = bool(clean) and self.state.show_subtitles and self.state.mic_enabled
        self.subtitle_panel.setVisible(visible)
        self._position_subtitle()
        if visible:
            self._subtitle_hide_timer.start()
        else:
            self._subtitle_hide_timer.stop()
        if clean and self.state.mic_enabled and self.state.show_subtitles:
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
            self.state.camera_enabled = True
            self.bottom_controls.camera_button.setChecked(True)
            self.bottom_controls._refresh_button_style()
            self._open_camera(self.state.camera_index)
            self._capture_timer.setInterval(self._camera_timer_interval_ms)
            if self._camera is not None and self._camera.isOpened():
                self._capture_timer.start()
            else:
                self._show_placeholder_frame()
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

        if not self._open_video(path):
            self.status_chip.setText("Video open failed")
            return

        self._source_mode = "video"
        self._video_path = path
        self._video_paused = False
        self._video_ended = False
        self._analysis_busy = False
        self._cached_results = []
        self._video_frame_index = 0
        self.player_controls.play_pause_button.setText("⏸")
        self.player_controls.setVisible(True)
        self.bottom_controls.camera_button.setChecked(False)
        self.bottom_controls._refresh_button_style()
        self._capture_timer.setInterval(self._video_timer_interval_ms)
        self._capture_timer.start()
        self.status_chip.setText(f"Video: {Path(path).name}")
        self._apply_layout()

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
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_camera = dialog.selected_camera()
        new_microphone = dialog.selected_microphone()
        self.state.show_subtitles = dialog.subtitles_enabled()
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

        self.status_chip.setText(
            f"Camera {self.state.camera_index} | Mic {self.state.mic_device if self.state.mic_device is not None else 'default'}"
        )
        self.set_subtitle("Devices updated")

    def _default_model_path(self) -> Optional[str]:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "checkpoints_rafdb" / "best_emotion_model.pt",
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

    def _is_valid_vosk_model_dir(self, path: Path) -> bool:
        required = ["am", "conf", "graph"]
        return path.exists() and path.is_dir() and all((path / item).exists() for item in required)

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
            self.set_subtitle(clean)

    def _on_voice_final(self, text: str) -> None:
        clean = text.strip()
        if clean:
            self.set_subtitle(clean)

    def _on_voice_error(self, text: str) -> None:
        self.status_chip.setText("Mic error")
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