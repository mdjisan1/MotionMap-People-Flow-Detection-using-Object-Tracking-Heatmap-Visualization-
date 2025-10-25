import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import os

# ---------- CONFIG ----------
VIDEO_PATH = "people-walking.mp4"
MODEL_PATH = "yolov8n.pt"
SLOWDOWN_MS = 1
TRACKER_YAML = "botsort.yaml"
OUTPUT_HEATMAP = "heatmap.png"
OUTPUT_VIDEO = "output_video.mp4"   # <--- NEW
MIN_CONFIDENCE = 0.35
HEATMAP_RADIUS = 30
HEATMAP_BLUR = 31

LINE_TOP_FRAC = 0.36
LINE_BOTTOM_FRAC = 0.64
PERSON_CLASS_ID = 0

def draw_text(img, text, org, scale=1.0, color=(255,255,255), thickness=2, bgcolor=(0,0,0)):
    x,y = org
    cv2.putText(img, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, bgcolor, thickness+2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)

def main():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print("Failed to load YOLO model:", e)
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video:", VIDEO_PATH)
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return
    H, W = frame.shape[:2]

    # compute line positions
    line_in_y = int(LINE_TOP_FRAC * H)
    line_out_y = int(LINE_BOTTOM_FRAC * H)
    line_x1, line_x2 = 0, W

    # ----------- NEW: setup video writer -----------
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))
    # ------------------------------------------------

    heatmap_acc = np.zeros((H, W), dtype=np.float32)
    in_count = 0
    out_count = 0
    already_counted_in = set()
    already_counted_out = set()
    track_history = {}
    frame_idx = 0
    use_tracker = True if TRACKER_YAML else False

    print("Starting processing... press 'q' to quit early")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        try:
            if use_tracker:
                results = model.track(frame, persist=True, tracker=TRACKER_YAML, verbose=False)[0]
            else:
                results = model(frame)[0]
        except Exception as e:
            results = model(frame)[0]

        boxes = getattr(results, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            display = frame.copy()
            draw_text(display, f"In: {in_count}  Out: {out_count}", (10,40), scale=1.2, color=(0,255,0))
            cv2.imshow("People Flow", cv2.resize(display, (960,540)))
            out_writer.write(frame)  # still save even if no detections
            if cv2.waitKey(SLOWDOWN_MS) & 0xFF == ord('q'):
                break
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                track_ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                track_ids = None
        if track_ids is None:
            track_ids = np.arange(len(xyxy), dtype=int)

        for i, (b, c, confscore, tid) in enumerate(zip(xyxy, cls, conf, track_ids)):
            if confscore < MIN_CONFIDENCE:
                continue
            if c != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, b)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(heatmap_acc, (cx, cy), HEATMAP_RADIUS, 1.0, -1)

            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 2:
                track_history[tid] = track_history[tid][-2:]

            if len(track_history[tid]) == 2:
                prev, curr = track_history[tid]
                if (prev[1] < line_in_y) and (curr[1] >= line_in_y) and (tid not in already_counted_in):
                    in_count += 1
                    already_counted_in.add(tid)
                if (prev[1] > line_out_y) and (curr[1] <= line_out_y) and (tid not in already_counted_out):
                    out_count += 1
                    already_counted_out.add(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
            draw_text(frame, f"ID {tid}", (x1, y1-8), scale=0.5, color=(0,255,0))

        # Draw lines + counts
        cv2.line(frame, (line_x1, line_in_y), (line_x2, line_in_y), (255,0,0), 2)
        cv2.line(frame, (line_x1, line_out_y), (line_x2, line_out_y), (0,0,255), 2)
        draw_text(frame, f"In: {in_count}", (10,40), scale=1.5, color=(0,255,0))
        draw_text(frame, f"Out: {out_count}", (10,90), scale=1.5, color=(0,0,255))

        # ---------- Write frame to video -----------
        out_writer.write(frame)
        # ------------------------------------------

        cv2.imshow("People Flow", cv2.resize(frame, (960,540)))
        if cv2.waitKey(SLOWDOWN_MS) & 0xFF == ord('q'):
            break

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"Frame {frame_idx} processed. In={in_count} Out={out_count}. FPS ~ {frame_idx/elapsed:.1f}")

    # ---------- Release writer ----------
    out_writer.release()
    print(f"Saved output video to {OUTPUT_VIDEO}")
    # -----------------------------------

    # ---------- Final heatmap ----------
    heat = heatmap_acc.copy()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.GaussianBlur(heat, (HEATMAP_BLUR, HEATMAP_BLUR), 0)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    try:
        base_frame = frame.copy()
    except Exception:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, base_frame = cap.read()
    overlayed = cv2.addWeighted(base_frame, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(OUTPUT_HEATMAP, overlayed)
    print(f"Saved final heatmap to {OUTPUT_HEATMAP}")
    cv2.imshow("Final Heatmap Overlay", cv2.resize(overlayed, (960,540)))
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
