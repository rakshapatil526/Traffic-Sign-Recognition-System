"""
Traffic Sign Recognition System - Detection Engine
Implements OpenCV candidate RoI localization + Deep CNN classification + NMS + HUD Annotation.
Also provides plug-and-play YOLO ONNX neural network detection.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.config import TRAFFIC_SIGN_CLASSES, CATEGORY_COLORS, get_class_info
from src.model_cnn import classify_crop

# Detection parameters
MIN_SIGN_AREA = 400       # Minimum bounding box area in pixels (20x20)
MAX_SIGN_AREA_RATIO = 0.4 # Maximum bounding box area relative to image
ASPECT_RATIO_MIN = 0.65   # Width / Height ratio minimum
ASPECT_RATIO_MAX = 1.45   # Width / Height ratio maximum

class TrafficSignDetector:
    """
    Two-stage Traffic Sign Detection and Recognition System:
    Stage 1: Multi-scale Color-Geometric RoI candidate proposal (or YOLO ONNX)
    Stage 2: Deep Convolutional Neural Network (CNN) Feature Classification
    Stage 3: Non-Maximum Suppression (NMS) and HUD rendering
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.conf_thresh = confidence_threshold
        self.yolo_net = None
        self._check_yolo_onnx()

    def _check_yolo_onnx(self):
        """Check if an ONNX YOLO model file is present in models directory."""
        onnx_candidates = [
            os.path.join(BASE_DIR, "models", "traffic_sign_yolo.onnx"),
            os.path.join(BASE_DIR, "models", "yolov8n.onnx"),
            os.path.join(BASE_DIR, "models", "yolov5s.onnx")
        ]
        for path in onnx_candidates:
            if os.path.exists(path):
                try:
                    self.yolo_net = cv2.dnn.readNetFromONNX(path)
                    print(f"Loaded YOLO ONNX detector from {path}")
                    break
                except Exception as e:
                    print(f"Note: Could not initialize ONNX model from {path}: {e}")

    def extract_candidate_regions(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Stage 1: Generates candidate bounding boxes (x, y, w, h) for traffic signs
        using multi-color space segmentation (Red, Blue, Yellow) and contour geometry.
        """
        h_img, w_img = img_bgr.shape[:2]
        total_area = h_img * w_img
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Red masks (covers hue wrap-around 0-12 and 165-180)
        mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        
        # 2. Blue mask (covers mandatory turn / roundabout signs)
        mask_blue = cv2.inRange(hsv, np.array([95, 80, 50]), np.array([135, 255, 255]))
        
        # 3. Yellow/Amber mask (covers caution / warning signs)
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 70]), np.array([36, 255, 255]))
        
        # Combine candidate masks
        combined_mask = cv2.bitwise_or(mask_red, mask_blue)
        combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
        
        # Morphological operations to bridge broken edges and fill interior gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_SIGN_AREA or area > (total_area * MAX_SIGN_AREA_RATIO):
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)
            
            if ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX:
                # Add tiny padding (4%) to capture complete border without background intrusion
                pad_x = max(1, int(w * 0.04))
                pad_y = max(1, int(h * 0.04))
                x_pad = max(0, x - pad_x)
                y_pad = max(0, y - pad_y)
                w_pad = min(w_img - x_pad, w + 2 * pad_x)
                h_pad = min(h_img - y_pad, h + 2 * pad_y)
                candidates.append((x_pad, y_pad, w_pad, h_pad))
                
        return candidates

    def detect_and_classify(
        self,
        img_bgr: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Full Detection & Recognition Pipeline:
        1. Propose candidate RoIs
        2. Deep CNN classification
        3. Non-Maximum Suppression (NMS)
        4. Package detailed metadata
        """
        thresh = (confidence_threshold or self.conf_thresh) * 100.0
        h_img, w_img = img_bgr.shape[:2]
        
        # Stage 1: RoI Proposals
        candidates = self.extract_candidate_regions(img_bgr)
        
        # If no candidates found with color mask, try adaptive edge/contour proposal
        if not candidates:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 60, 180)
            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > MIN_SIGN_AREA and area < (h_img * w_img * 0.3):
                    x, y, w, h = cv2.boundingRect(cnt)
                    ar = float(w) / float(h)
                    if 0.8 <= ar <= 1.25:
                        candidates.append((x, y, w, h))
                        
        if not candidates:
            return []

        # Convert image to RGB for deep CNN classification
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        boxes = []
        confidences = []
        class_ids = []
        crops = []

        # Stage 2: Deep CNN Inference
        for (x, y, w, h) in candidates:
            crop = img_rgb[y:y+h, x:x+w]
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                continue
                
            cid, conf, _ = classify_crop(crop)
            
            if conf >= thresh:
                boxes.append([x, y, w, h])
                confidences.append(float(conf / 100.0))
                class_ids.append(cid)
                crops.append(crop)

        if not boxes:
            return []

        # Stage 3: Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=float(thresh / 100.0),
            nms_threshold=0.35
        )

        detections = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                x, y, w, h = boxes[idx]
                cid = class_ids[idx]
                conf = confidences[idx] * 100.0
                crop = crops[idx]
                info = get_class_info(cid)
                
                detections.append({
                    "class_id": cid,
                    "name": info["name"],
                    "category": info["category"],
                    "confidence": round(conf, 1),
                    "meaning": info["meaning"],
                    "instruction": info["instruction"],
                    "penalty_risk": info["penalty_risk"],
                    "shape": info["shape"],
                    "priority": info["priority"],
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "crop_rgb": crop
                })

        return detections

    def draw_detections(
        self,
        img_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_hud: bool = True
    ) -> np.ndarray:
        """
        Renders futuristic, clean ADAS HUD bounding boxes and labels onto the image.
        """
        annotated = img_bgr.copy()
        
        for det in detections:
            x, y, w, h = det["bbox"]
            cat = det["category"]
            name = det["name"]
            conf = det["confidence"]
            
            color_meta = CATEGORY_COLORS.get(cat, {"bgr": (0, 255, 128), "icon": "🔍"})
            color_bgr = color_meta["bgr"]
            
            # 1. Main bounding box rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2, cv2.LINE_AA)
            
            # 2. Tech HUD corner accents
            corner_len = min(16, w // 4, h // 4)
            thick = 3
            # Top-left
            cv2.line(annotated, (x, y), (x + corner_len, y), color_bgr, thick)
            cv2.line(annotated, (x, y), (x, y + corner_len), color_bgr, thick)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color_bgr, thick)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color_bgr, thick)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color_bgr, thick)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color_bgr, thick)
            # Bottom-right
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color_bgr, thick)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color_bgr, thick)

            # 3. Label tag banner
            label = f"{name} | {conf:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            
            tag_y1 = max(0, y - th - 12)
            tag_y2 = y
            tag_x1 = x
            tag_x2 = min(annotated.shape[1], x + tw + 14)
            
            # Dark glass background for tag
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color_bgr, 1)
            
            # Text
            text_pos = (tag_x1 + 6, y - 5)
            cv2.putText(annotated, label, text_pos, font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        return annotated
