"""
=============================================================
 Netflix Intelligence System
 MODULE: Computer Vision — Thumbnail / Poster Analyser
 Operations: Color analysis, mood detection, text density,
             brightness, contrast, dominant colours, filters
=============================================================
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PosterAnalysisResult:
    """Results of a single CV analysis on a poster/thumbnail."""
    width: int
    height: int
    channels: int
    brightness: float          # 0–255
    contrast: float            # std-dev of grayscale
    dominant_colors: list      # top-3 hex colours
    color_mood: str            # Dark / Bright / Warm / Cool / Neutral
    edge_density: float        # 0–1  (text/detail richness)
    face_count: int            # faces detected (Haar)
    genre_hint: str            # inferred genre tone
    annotated_frame: Optional[np.ndarray] = field(default=None, repr=False)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _dominant_colors(img_rgb: np.ndarray, k: int = 3) -> list[str]:
    """K-means dominant colour extraction."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    # Subsample for speed
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    # Sort by frequency
    counts = np.bincount(labels.flatten())
    order  = np.argsort(-counts)
    return [_rgb_to_hex(*centers[i]) for i in order]


def _color_mood(img_rgb: np.ndarray) -> str:
    """Classify overall mood from average HSV."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    avg_v = hsv[:, :, 2].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_h = hsv[:, :, 0].mean()

    if avg_v < 60:
        return "🌑 Dark / Thriller"
    if avg_v > 180 and avg_s < 50:
        return "⬜ Bright / Clean"
    if 0 <= avg_h <= 15 or avg_h >= 165:          # red hues
        return "🔴 Warm / Drama"
    if 90 <= avg_h <= 130 and avg_s > 60:          # blue hues
        return "🔵 Cool / Sci-Fi"
    if 30 <= avg_h <= 90 and avg_s > 60:           # green/yellow
        return "🟢 Natural / Adventure"
    return "⚪ Neutral / General"


def _genre_hint(mood: str, edge_density: float, face_count: int) -> str:
    if "Dark" in mood or "Thriller" in mood:
        return "🎭 Thriller / Horror / Crime"
    if face_count >= 3:
        return "👥 Drama / Rom-Com / Reality"
    if face_count == 0 and edge_density > 0.15:
        return "🚀 Sci-Fi / Animation / Documentary"
    if "Warm" in mood and face_count >= 1:
        return "❤️ Romance / Drama"
    if "Natural" in mood:
        return "🌿 Adventure / Nature / Documentary"
    if edge_density > 0.20:
        return "💥 Action / Thriller"
    return "🎬 General / Mixed"


# ── Main Analyser ─────────────────────────────────────────────────────────────
class NetflixPosterAnalyser:
    """
    Analyse any uploaded image as if it were a Netflix poster / thumbnail.
    No deep-learning required — uses OpenCV classical methods.
    """

    def __init__(self):
        # Load Haar face cascade (bundled with OpenCV)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def analyse(self, frame_bgr: np.ndarray) -> PosterAnalysisResult:
        """
        Args:
            frame_bgr: image loaded with cv2.imdecode (BGR uint8)
        Returns:
            PosterAnalysisResult with all metrics
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w, ch = img_rgb.shape

        # Brightness & contrast
        brightness = float(gray.mean())
        contrast   = float(gray.std())

        # Dominant colours
        dom_colors = _dominant_colors(img_rgb)

        # Mood
        mood = _color_mood(img_rgb)

        # Edge density (Canny)
        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(edges.mean()) / 255.0

        # Face detection
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        face_count = len(faces)

        # Genre hint
        genre = _genre_hint(mood, edge_density, face_count)

        # Annotate — draw face boxes
        annotated = frame_bgr.copy()
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), (0, 255, 100), 2)
            cv2.putText(annotated, "Face", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

        return PosterAnalysisResult(
            width=w, height=h, channels=ch,
            brightness=round(brightness, 1),
            contrast=round(contrast, 1),
            dominant_colors=dom_colors,
            color_mood=mood,
            edge_density=round(edge_density, 4),
            face_count=face_count,
            genre_hint=genre,
            annotated_frame=annotated,
        )

    # ── Filter gallery ──────────────────────────────────────────────────────
    @staticmethod
    def apply_filters(img_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Return dict of named filter outputs (all RGB or grayscale)."""
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        filters = {}

        filters["Original"]           = img_rgb
        filters["Grayscale"]          = gray
        filters["Edge Detection"]     = cv2.Canny(gray, 80, 180)
        filters["Gaussian Blur"]      = cv2.GaussianBlur(img_rgb, (15, 15), 0)
        filters["Sharpen"]            = cv2.filter2D(
                                            img_rgb, -1,
                                            np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                                        )
        filters["Emboss"]             = cv2.filter2D(
                                            gray, -1,
                                            np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
                                        )
        filters["Invert"]             = cv2.bitwise_not(img_rgb)

        # Sepia
        k = np.array([[0.272,0.534,0.131],
                      [0.349,0.686,0.168],
                      [0.393,0.769,0.189]])
        filters["Sepia"]              = np.clip(img_rgb @ k.T, 0, 255).astype(np.uint8)

        # Netflix-style red tint
        red_tint = img_rgb.copy()
        red_tint[:, :, 0] = np.clip(red_tint[:, :, 0].astype(int) + 60, 0, 255)
        red_tint[:, :, 1] = (red_tint[:, :, 1] * 0.75).astype(np.uint8)
        red_tint[:, :, 2] = (red_tint[:, :, 2] * 0.75).astype(np.uint8)
        filters["Netflix Red Tint"]   = red_tint

        # High contrast
        filters["High Contrast"]      = cv2.convertScaleAbs(img_rgb, alpha=1.8, beta=-60)

        return filters

    # ── Pixel stats ─────────────────────────────────────────────────────────
    @staticmethod
    def pixel_stats(img_rgb: np.ndarray) -> list[dict]:
        return [
            {
                "Channel": ch,
                "Mean": round(img_rgb[:, :, i].mean(), 2),
                "Std Dev": round(img_rgb[:, :, i].std(), 2),
                "Min": int(img_rgb[:, :, i].min()),
                "Max": int(img_rgb[:, :, i].max()),
            }
            for i, ch in enumerate(["Red", "Green", "Blue"])
        ]
