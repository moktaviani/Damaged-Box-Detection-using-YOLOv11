"""
Damage/package detection
"""
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class DamageDetector:
    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        agnostic_nms: bool = False,
        device: Optional[str] = None,
    ):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}\n"
                f"Download 'best.pt' from your Colab training run "
                f"(damaged_package_runs/train/weights/best.pt) and place it "
                f"next to this script, or pass the correct path with --weights."
            )

        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.device = device
        self.class_names = self.model.names 

    def predict_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()  

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class": self.class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )
        return annotated, detections