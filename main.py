"""
python main.py --mode camera --weights best.pt
python main.py --mode image  --weights best.pt --source photo.jpg
python main.py --mode image  --weights best.pt --source my_photos_folder
python main.py --mode video  --weights best.pt --source clip.mp4 --output outputs
"""
import argparse
from pathlib import Path

import cv2

from detector import DamageDetector
from utils import FPSCounter, ResourceMonitor, draw_resources, make_output_path, print_detections

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Damage box detection (YOLO)")
    parser.add_argument("--mode", required=True, choices=["camera", "image", "video"])
    parser.add_argument("--weights", default="best.pt", help="Path to trained best.pt")
    parser.add_argument(
        "--source",
        default=None,
        help="Image/folder path (mode=image) or video file path (mode=video). Ignored for camera.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for mode=camera")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (lower = catch more, but noisier)")
    parser.add_argument("--iou", type=float, default=0.45,
                         help="NMS IoU threshold. Raise this (e.g. 0.6-0.7) if nearby/overlapping "
                              "boxes are getting merged into one in crowded scenes.")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution. Raise this (e.g. 960, 1280) for high-res frames "
                              "with many small/distant boxes, at the cost of speed.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections to keep per frame")
    parser.add_argument("--agnostic-nms", action="store_true",
                         help="Run NMS across all classes together instead of per-class. Useful if "
                              "Damaged/Normal boxes on the same physical package are suppressing each other.")
    parser.add_argument("--device", default=None, help="'cpu', '0' for first GPU, etc. Default: auto")
    parser.add_argument("--output", default="outputs", help="Folder to save results")
    parser.add_argument("--no-show", action="store_true", help="Don't open a display window (e.g. on a headless server)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    return parser.parse_args()


def run_camera(detector: DamageDetector, args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    writer = None
    if not args.no_save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        out_path = make_output_path(args.output, "camera_session", ".mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        print(f"Recording to: {out_path}")

    fps_counter = FPSCounter()
    monitor = ResourceMonitor(interval=5)
    print("Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            annotated, detections = detector.predict_frame(frame)
            fps = fps_counter.tick()
            res = monitor.sample(fps=fps)
            draw_resources(annotated, res)

            if writer is not None:
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("Damage Detection - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def run_image(detector: DamageDetector, args):
    if not args.source:
        raise ValueError("--source is required for mode=image (a file or a folder)")
    source_path = Path(args.source)

    if source_path.is_dir():
        image_paths = sorted(p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif source_path.is_file():
        image_paths = [source_path]
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    if not image_paths:
        print("No images found.")
        return

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Could not read image: {img_path}")
            continue

        annotated, detections = detector.predict_frame(frame)
        print_detections(detections, label=img_path.name)

        if not args.no_save:
            out_path = make_output_path(args.output, img_path.name, img_path.suffix)
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved: {out_path}")

        if not args.no_show:
            cv2.imshow(img_path.name, annotated)
            print("Press any key to continue, 'q' to stop early.")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break


def run_video(detector: DamageDetector, args):
    if not args.source:
        raise ValueError("--source is required for mode=video (a video file path)")
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if not args.no_save:
        out_path = make_output_path(args.output, source_path.name, ".mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        print(f"Saving annotated video to: {out_path}")

    fps_counter = FPSCounter()
    monitor = ResourceMonitor(interval=5)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            annotated, detections = detector.predict_frame(frame)
            fps = fps_counter.tick()
            res = monitor.sample(fps=fps)
            draw_resources(annotated, res)

            if detections:
                print_detections(detections, label=f"frame {frame_idx}/{total_frames}")

            if writer is not None:
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("Damage Detection - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    detector = DamageDetector(
        args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        device=args.device,
    )
    print(f"Loaded model. Classes: {detector.class_names}")

    if args.mode == "camera":
        run_camera(detector, args)
    elif args.mode == "image":
        run_image(detector, args)
    elif args.mode == "video":
        run_video(detector, args)


if __name__ == "__main__":
    main()