# """
# Drowsiness Detection 
# Face + Eye (PyTorch MobileNetV2) + Yawn (TF RegNetX)

# Alert Logic:
#   OPEN   + NO_YAWN = NORMAL            (no alert)
#   CLOSED + NO_YAWN = DROWSY            (alert after eye_closed_streak frames)
#   OPEN   + YAWN    = DROWSY            (alert after yawn_streak_needed frames)
#   CLOSED + YAWN    = DROWSY IMMEDIATE  (alert immediately)
#   NO FACE          = FACE NOT VISIBLE  (alert after no_face_alert_frames)

#     eye-model drowsy_mobilenetv2.pth
#     yawn-model yawn_model.keras
# """

# import cv2
# import numpy as np
# import tensorflow as tf
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import argparse
# import time
# import os
# import sys
# from collections import deque
# from datetime import datetime

# try:
#     import pygame
#     pygame.mixer.init()
#     SOUND_AVAILABLE = True
# except ImportError:
#     SOUND_AVAILABLE = False


# EYE_CLOSED_CLASS_INDEX = 0


# CONFIG = {
#     "yawn_model_path":       "yawn_model.keras",
#     "eye_model_path":        "drowsy_mobilenetv2.pth",
#     "yawn_img_size":         (224, 224),
#     "eye_img_size":          (224, 224),
#     "yawn_class_index":      1,
#     "yawn_thresh":           0.50,
#     "eye_closed_thresh":     0.65,
#     "yawn_streak_needed":    3,
#     "eye_closed_streak":     15,
#     # [FIX BUG 2] How many consecutive no-face frames before alerting
#     "no_face_alert_frames":  30,
#     "alert_cooldown_sec":    8,
#     "yawn_smooth_window":    6,
#     "eye_smooth_window":     8,
#     "face_scale_factor":     1.1,
#     "face_min_neighbors":    5,
#     "face_min_size":         (80, 80),
#     "font":                  cv2.FONT_HERSHEY_SIMPLEX,
#     "display_width":         1280,
#     "display_height":        720,
#     "alert_sound_path":      "alert.wav",
# }

# COLOR = {
#     "green":  (34, 197, 94),
#     "red":    (50,  50, 220),
#     "yellow": (0,  200, 255),
#     "white":  (255, 255, 255),
#     "gray":   (120, 120, 120),
#     "dark":   (20,  20,  30),
#     "cyan":   (255, 200,   0),
#     "orange": (0,  140, 255),
#     "accent": (0,  165, 255),
# }



# def load_yawn_model(path):
#     if not os.path.exists(path):
#         print(f"[WARN] Yawn model not found: {path}")
#         return None
#     print(f"[INFO] Loading yawn model: {path}")
#     try:
#         model = tf.keras.models.load_model(path, compile=False)
#     except Exception:
#         model = tf.keras.models.load_model(path)
#     model.compile(
#         optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     print(f"[INFO] Yawn model loaded. Input: {model.input_shape}")
#     return model


# def load_eye_model(path):
#     if not os.path.exists(path):
#         print(f"[WARN] Eye model not found: {path}")
#         return None
#     print(f"[INFO] Loading eye model: {path}")
#     model = models.mobilenet_v2(weights=None)
#     model.classifier[1] = nn.Linear(model.last_channel, 2)
#     state_dict = torch.load(path, map_location="cpu", weights_only=False)
#     model.load_state_dict(state_dict)
#     model.eval()
#     print("[INFO] Eye model loaded.")
#     return model


# def load_face_detector():
#     det = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     if det.empty():
#         print("[ERROR] Face detector failed.")
#         sys.exit(1)
#     print("[INFO] Face detector loaded.")
#     return det


# def load_eye_cascade():
#     det = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_eye.xml")
#     if det.empty():
#         return None
#     print("[INFO] Eye cascade loaded.")
#     return det


# def load_mouth_cascade():
#     det = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_smile.xml")
#     if det.empty():
#         return None
#     print("[INFO] Mouth cascade loaded.")
#     return det


# def detect_faces(gray, detector):
#     faces = detector.detectMultiScale(
#         gray,
#         scaleFactor=CONFIG["face_scale_factor"],
#         minNeighbors=CONFIG["face_min_neighbors"],
#         minSize=CONFIG["face_min_size"],
#     )
#     return list(faces) if len(faces) > 0 else []


# def detect_eyes(face_gray, face_x, face_y, face_w, face_h, eye_cascade):
#     upper = face_gray[:int(face_h * 0.60), :]
#     eyes = eye_cascade.detectMultiScale(
#         upper, scaleFactor=1.1, minNeighbors=8,
#         minSize=(face_w // 6, face_h // 8))
#     return [(face_x + ex, face_y + ey, ew, eh)
#             for (ex, ey, ew, eh) in eyes[:2]]


# def detect_mouth(face_gray, face_x, face_y, face_w, face_h, mouth_cascade):
#     lower_start = int(face_h * 0.45)
#     lower = face_gray[lower_start:, :]
#     mouths = mouth_cascade.detectMultiScale(
#         lower, scaleFactor=1.07, minNeighbors=8,
#         minSize=(face_w // 4, face_h // 8))
#     if len(mouths) == 0:
#         return None
#     mouths = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)
#     mx, my, mw, mh = mouths[0]
#     return (face_x + mx, face_y + lower_start + my, mw, mh)


# def preprocess_keras(roi):
#     img = cv2.resize(roi, CONFIG["yawn_img_size"])
#     if len(img.shape) == 2:
#         img = np.stack([img, img, img], axis=-1)
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0)


# eye_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(CONFIG["eye_img_size"]),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])


# def preprocess_pytorch(roi_bgr):
#     rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
#     return eye_transform(rgb).unsqueeze(0)


# def eye_closed_confidence(model, roi_bgr, debug=False):
#     """Returns probability [0..1] that eyes are CLOSED."""
#     with torch.no_grad():
#         probs = torch.softmax(model(preprocess_pytorch(roi_bgr)), dim=1)[0]
#         p0 = float(probs[0])
#         p1 = float(probs[1])

#     if debug:
#         print(f"  [EYE DEBUG] p[0]={p0:.3f}  p[1]={p1:.3f}  "
#               f"using index={EYE_CLOSED_CLASS_INDEX}  "
#               f"closed_conf={float(probs[EYE_CLOSED_CLASS_INDEX]):.3f}")

#     return float(probs[EYE_CLOSED_CLASS_INDEX])



# class AlertSystem:
#     def __init__(self):
#         self.last_alert_time = 0
#         self.alert_active    = False
#         self.alert_reason    = ""
#         self.alert_count     = 0        
#         self.sound = None
#         if SOUND_AVAILABLE and os.path.exists(CONFIG["alert_sound_path"]):
#             try:
#                 self.sound = pygame.mixer.Sound(CONFIG["alert_sound_path"])
#             except Exception:
#                 pass

#     def trigger(self, reason=""):
#         now = time.time()
#         if now - self.last_alert_time >= CONFIG["alert_cooldown_sec"]:
#             self.last_alert_time = now
#             self.alert_active    = True
#             self.alert_reason    = reason
#             self.alert_count    += 1      
#             print(f"[ALERT] {reason} at {datetime.now().strftime('%H:%M:%S')}")
#             if self.sound:
#                 self.sound.play()

#     def reset(self):
#         self.alert_active = False

# # ═══════════════════════════════════════════════════════════════════
# #  STATS TRACKER
# # ═══════════════════════════════════════════════════════════════════

# class StatsTracker:
#     def __init__(self):
#         self.total_frames        = 0
#         self.yawn_frames         = 0
#         self.eye_closed_frames   = 0
#         self.consecutive_yawns   = 0
#         self.consecutive_closed  = 0
#         self.consecutive_no_face = 0   
#         self.start_time          = time.time()
#         self.fps_buffer          = deque(maxlen=30)
#         self.yawn_buf            = deque(maxlen=CONFIG["yawn_smooth_window"])
#         self.eye_buf             = deque(maxlen=CONFIG["eye_smooth_window"])
#         self.prev_time           = time.time()

#     def update_fps(self):
#         now = time.time()
#         fps = 1.0 / max(now - self.prev_time, 1e-6)
#         self.fps_buffer.append(fps)
#         self.prev_time = now
#         return float(np.mean(self.fps_buffer))

#     def smooth_yawn(self, c):
#         self.yawn_buf.append(c)
#         return float(np.mean(self.yawn_buf))

#     def smooth_eye(self, c):
#         self.eye_buf.append(c)
#         return float(np.mean(self.eye_buf))

#     def duration(self):
#         s = int(time.time() - self.start_time)
#         return f"{s//60:02d}:{s%60:02d}"

#     @property
#     def yawn_rate(self):
#         return self.yawn_frames / max(self.total_frames, 1) * 100

#     @property
#     def eye_closed_rate(self):
#         return self.eye_closed_frames / max(self.total_frames, 1) * 100



# def overlay(img, pt1, pt2, color, alpha=0.65):
#     ov = img.copy()
#     cv2.rectangle(ov, pt1, pt2, color, -1)
#     cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


# def draw_bar(frame, x, y, w, h, val, label, color):
#     cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR["gray"], -1)
#     cv2.rectangle(frame, (x, y), (x + int(w * val), y + h), color, -1)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR["white"], 1)
#     cv2.putText(frame, f"{label}: {val*100:.0f}%",
#                 (x, y - 5), CONFIG["font"], 0.4, COLOR["white"], 1, cv2.LINE_AA)


# def draw_hud(frame, stats, alert, yawn_label, yawn_conf,
#              eye_label, eye_conf, num_faces, fps, is_drowsy):
#     h, w = frame.shape[:2]
#     f = CONFIG["font"]

#     # Top bar
#     overlay(frame, (0, 0), (w, 52), COLOR["dark"], 0.8)
#     cv2.putText(frame, "DROWSINESS DETECTION SYSTEM",
#                 (12, 34), f, 0.7, COLOR["accent"], 2, cv2.LINE_AA)
#     cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 34), f, 0.6,
#                 COLOR["green"] if fps >= 15 else COLOR["yellow"], 1, cv2.LINE_AA)
#     cv2.putText(frame, stats.duration(), (w - 240, 34), f, 0.6,
#                 COLOR["white"], 1, cv2.LINE_AA)

#     if num_faces == 0:
#         status_text  = "NO FACE"
#         status_color = COLOR["yellow"]
#     else:
#         status_text  = "DROWSY" if is_drowsy else "NORMAL"
#         status_color = COLOR["red"] if is_drowsy else COLOR["green"]
#     overlay(frame, (w - 160, 58), (w - 10, 100), status_color, 0.85)
#     cv2.putText(frame, status_text, (w - 150, 90), f, 0.75,
#                 COLOR["white"], 2, cv2.LINE_AA)

#     if alert.alert_active:
#         pulse = int(abs(np.sin(time.time() * 4)) * 180)
#         overlay(frame, (0, 57), (w - 170, 115), (0, 0, 60 + pulse), 0.88)
#         cv2.putText(frame, f"!! ALERT: {alert.alert_reason} !!",
#                     (20, 97), f, 0.85, COLOR["white"], 2, cv2.LINE_AA)

#     px, py, pw = 10, h - 290, 290
#     overlay(frame, (px, py), (px + pw, py + 275), COLOR["dark"], 0.75)

#     yc = COLOR["red"] if yawn_label == "YAWN" else COLOR["green"]
#     cv2.putText(frame, "YAWN",     (px+10, py+22),  f, 0.45, COLOR["gray"], 1, cv2.LINE_AA)
#     cv2.putText(frame, yawn_label, (px+10, py+52),  f, 0.9,  yc, 2, cv2.LINE_AA)
#     draw_bar(frame, px+10, py+62, pw-20, 14, yawn_conf, "Conf", COLOR["red"])

#     ec = COLOR["red"] if eye_label == "CLOSED" else COLOR["green"]
#     cv2.putText(frame, "EYES",    (px+10, py+102), f, 0.45, COLOR["gray"], 1, cv2.LINE_AA)
#     cv2.putText(frame, eye_label, (px+10, py+132), f, 0.9,  ec, 2, cv2.LINE_AA)
#     draw_bar(frame, px+10, py+142, pw-20, 14, eye_conf, "Conf", COLOR["orange"])

#     cv2.putText(frame, f"Faces: {num_faces}   Alerts: {alert.alert_count}",
#                 (px+10, py+178), f, 0.45, COLOR["white"], 1, cv2.LINE_AA)
#     cv2.putText(frame, f"Yawn Rate:  {stats.yawn_rate:.1f}%",
#                 (px+10, py+198), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)
#     cv2.putText(frame, f"Eye Closed: {stats.eye_closed_rate:.1f}%",
#                 (px+10, py+216), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)
#     cv2.putText(frame, f"Frames: {stats.total_frames}",
#                 (px+10, py+234), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)

#     # Yawn streak
#     cv2.putText(frame,
#                 f"Yawn streak ({stats.consecutive_yawns}/{CONFIG['yawn_streak_needed']})",
#                 (px+10, py+255), f, 0.4, COLOR["gray"], 1, cv2.LINE_AA)
#     for i in range(CONFIG["yawn_streak_needed"]):
#         filled = i < stats.consecutive_yawns
#         cv2.circle(frame, (px + 15 + i * 22, h - 20), 7,
#                    COLOR["red"] if filled else COLOR["gray"], -1 if filled else 2)

#     bw    = pw - 20
#     ratio = min(stats.consecutive_closed / CONFIG["eye_closed_streak"], 1.0)
#     cv2.rectangle(frame, (px+10, h-45), (px+10+bw, h-32), COLOR["gray"], -1)
#     cv2.rectangle(frame, (px+10, h-45), (px+10+int(bw*ratio), h-32),
#                   COLOR["orange"], -1)
#     cv2.putText(frame,
#                 f"Eye closed: {stats.consecutive_closed}/{CONFIG['eye_closed_streak']}",
#                 (px+10, h-50), f, 0.38, COLOR["gray"], 1, cv2.LINE_AA)

#     if num_faces == 0:
#         nf_ratio = min(stats.consecutive_no_face / CONFIG["no_face_alert_frames"], 1.0)
#         bx = w - 220
#         by = h - 30
#         cv2.rectangle(frame, (bx, by), (bx+200, by+12), COLOR["gray"], -1)
#         cv2.rectangle(frame, (bx, by), (bx+int(200*nf_ratio), by+12),
#                       COLOR["yellow"], -1)
#         cv2.putText(frame,
#                     f"No face: {stats.consecutive_no_face}/{CONFIG['no_face_alert_frames']}",
#                     (bx, by-5), f, 0.4, COLOR["yellow"], 1, cv2.LINE_AA)


# def draw_face_box(frame, x, y, w, h, yawn_label, eye_label, yawn_conf):
#     is_drowsy = (yawn_label == "YAWN") or (eye_label == "CLOSED")
#     color = COLOR["red"] if is_drowsy else COLOR["green"]
#     L = 22
#     for (px, py), (dx, dy) in [
#         ((x, y), (1, 0)), ((x, y), (0, 1)),
#         ((x+w, y), (-1, 0)), ((x+w, y), (0, 1)),
#         ((x, y+h), (1, 0)), ((x, y+h), (0, -1)),
#         ((x+w, y+h), (-1, 0)), ((x+w, y+h), (0, -1))
#     ]:
#         cv2.line(frame, (px, py), (px + dx*L, py + dy*L), color, 2)

#     tag = f"{yawn_label}  EYE:{eye_label}  {yawn_conf*100:.0f}%"
#     (tw, th), _ = cv2.getTextSize(tag, CONFIG["font"], 0.5, 1)
#     cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
#     cv2.putText(frame, tag, (x + 4, y - 5),
#                 CONFIG["font"], 0.5, COLOR["white"], 1, cv2.LINE_AA)

# #  MAIN LOOP
# def run_detection(source, yawn_model_path, eye_model_path, debug_eye=False):
#     yawn_model  = load_yawn_model(yawn_model_path)
#     eye_model   = load_eye_model(eye_model_path)
#     face_det    = load_face_detector()
#     eye_cascade = load_eye_cascade()
#     mouth_cas   = load_mouth_cascade()
#     alert       = AlertSystem()
#     stats       = StatsTracker()

#     print(f"\n[INFO] Eye closed class index = {EYE_CLOSED_CLASS_INDEX}")
#     print(f"[INFO] If eyes are still inverted, change EYE_CLOSED_CLASS_INDEX "
#           f"to {1 - EYE_CLOSED_CLASS_INDEX} at the top of this file.\n")
#     if debug_eye:
#         print("[DEBUG] --debug-eye active: raw probabilities printed each frame.\n")

#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open: {source}")
#         sys.exit(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["display_width"])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["display_height"])

#     print("\n[INFO] Running — Q=quit  S=screenshot  R=reset\n")
#     print(f"  Yawn model : {'loaded' if yawn_model else 'MISSING'}")
#     print(f"  Eye  model : {'loaded' if eye_model  else 'MISSING'}\n")

#     yawn_label = "NO_YAWN"
#     eye_label  = "OPEN"
#     yawn_conf  = 0.0
#     eye_conf   = 0.0
#     is_drowsy  = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         stats.total_frames += 1
#         fps  = stats.update_fps()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces     = detect_faces(gray, face_det)
#         num_faces = len(faces)

#         #  NO FACE DETECTED → alert, not just skip
#         if num_faces == 0:
#             stats.consecutive_yawns  = 0
#             stats.consecutive_closed = 0
#             stats.consecutive_no_face += 1
#             is_drowsy = False
#             yawn_label = "NO_YAWN"
#             eye_label  = "OPEN"
#             yawn_conf  = 0.0
#             eye_conf   = 0.0

#             if stats.consecutive_no_face >= CONFIG["no_face_alert_frames"]:
#                 alert.trigger("FACE NOT VISIBLE")
#                 stats.consecutive_no_face = 0   # reset counter, keep alert banner

#             cv2.putText(frame, "No face detected",
#                         (20, 90), CONFIG["font"], 0.7,
#                         COLOR["yellow"], 2, cv2.LINE_AA)

#         else:
#             # Face found -> reset no face counter, clear no face alert state
#             # Only reset alert banner when NORMAL
#             stats.consecutive_no_face = 0

#             areas = [(fw*fh, (fx, fy, fw, fh)) for (fx, fy, fw, fh) in faces]
#             areas.sort(reverse=True)
#             fx, fy, fw, fh = areas[0][1]

#             x1 = max(0, fx - int(fw * 0.08))
#             y1 = max(0, fy - int(fh * 0.08))
#             x2 = min(frame.shape[1], fx + fw + int(fw * 0.08))
#             y2 = min(frame.shape[0], fy + fh + int(fh * 0.08))

#             face_bgr  = frame[y1:y2, x1:x2]
#             face_gray = gray[y1:y2, x1:x2]

#             # ── EYE (PyTorch) 
#             raw_eye    = 0.0
#             eye_source = "none"   # for debug display
#             face_h_px  = y2 - y1
#             face_w_px  = x2 - x1

#             if eye_model is not None:
#                 eyes = detect_eyes(face_gray, x1, y1, face_w_px, face_h_px,
#                                    eye_cascade) if eye_cascade else []
#                 if eyes:
#                     eye_source = "cascade"
#                     confs = []
#                     for (ex, ey, ew, eh) in eyes:
#                         roi = frame[ey:ey+eh, ex:ex+ew]
#                         if roi.size == 0:
#                             continue
#                         c = eye_closed_confidence(eye_model, roi, debug=debug_eye)
#                         confs.append(c)
#                         col = (COLOR["red"] if c > CONFIG["eye_closed_thresh"]
#                                else COLOR["green"])
#                         cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), col, 1)
#                     if confs:
#                         raw_eye = max(confs)

#                 # Use top 40% of face 
#                 elif face_h_px >= 60 and face_w_px >= 60:
#                     eye_source = "geo-fallback"
#                     eye_y2 = y1 + int(face_h_px * 0.40)
#                     upper_roi = frame[y1:eye_y2, x1:x2]
#                     if upper_roi.size > 0:
#                         raw_eye = eye_closed_confidence(
#                             eye_model, upper_roi, debug=debug_eye)
#                     if debug_eye:
#                         cv2.rectangle(frame, (x1, y1), (x2, eye_y2),
#                                       COLOR["yellow"], 1)
#                         cv2.putText(frame, "eye-geo", (x1, y1 - 5),
#                                     CONFIG["font"], 0.35, COLOR["yellow"],
#                                     1, cv2.LINE_AA)

#                 else:
#                     eye_source = "skip"
#                     if debug_eye:
#                         print("  [EYE DEBUG] Face ROI too small — skipping eye check")

#             if debug_eye and eye_source != "cascade":
#                 print(f"  [EYE DEBUG] source={eye_source}  raw_eye={raw_eye:.3f}")

#             eye_conf  = stats.smooth_eye(raw_eye)
            
#             if eye_source in ("cascade", "geo-fallback"):
#                 eye_label = "CLOSED" if eye_conf >= CONFIG["eye_closed_thresh"] else "OPEN"
#             else:
#                 eye_label = "OPEN"

#             #  YAWN (TensorFlow)
#             raw_yawn = 0.0
#             if yawn_model is not None:
#                 mouth_input = None
#                 if mouth_cas is not None:
#                     m = detect_mouth(face_gray, x1, y1, x2-x1, y2-y1, mouth_cas)
#                     if m is not None:
#                         mx, my, mw, mh = m
#                         mx1 = max(0, mx - int(mw*0.3))
#                         my1 = max(0, my - int(mh*0.3))
#                         mx2 = min(frame.shape[1], mx+mw+int(mw*0.3))
#                         my2 = min(frame.shape[0], my+mh+int(mh*0.3))
#                         mouth_input = gray[my1:my2, mx1:mx2]
#                         cv2.rectangle(frame, (mx1, my1), (mx2, my2),
#                                       COLOR["cyan"], 2)
#                         cv2.putText(frame, "mouth", (mx1, my1-5),
#                                     CONFIG["font"], 0.38, COLOR["cyan"],
#                                     1, cv2.LINE_AA)

#                 inp = preprocess_keras(
#                     mouth_input if mouth_input is not None
#                     else cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
#                 raw_yawn = float(
#                     yawn_model.predict(inp, verbose=0)[0][CONFIG["yawn_class_index"]])

#             yawn_conf  = stats.smooth_yawn(raw_yawn)
#             yawn_label = "YAWN" if yawn_conf >= CONFIG["yawn_thresh"] else "NO_YAWN"

#             #  UPDATE STREAKS
#             if yawn_label == "YAWN":
#                 stats.yawn_frames       += 1
#                 stats.consecutive_yawns += 1
#             else:
#                 stats.consecutive_yawns = 0

#             if eye_label == "CLOSED":
#                 stats.eye_closed_frames  += 1
#                 stats.consecutive_closed += 1
#             else:
#                 stats.consecutive_closed = 0

#             #  ALERT LOGIC
#             #  OPEN  + NO_YAWN = NORMAL            → reset
#             #  CLOSED + NO_YAWN = DROWSY            → alert after streak
#             #  OPEN  + YAWN    = DROWSY             → alert after streak
#             #  CLOSED + YAWN   = DROWSY IMMEDIATE   → alert now
            
#             is_drowsy = (eye_label == "CLOSED") or (yawn_label == "YAWN")

#             if eye_label == "OPEN" and yawn_label == "NO_YAWN":
#                 stats.consecutive_yawns  = 0
#                 stats.consecutive_closed = 0
#                 alert.reset()   # [FIX BUG 4] only reset when confirmed NORMAL

#             elif eye_label == "CLOSED" and yawn_label == "YAWN":
#                 alert.trigger("EYES CLOSED + YAWNING")
#                 stats.consecutive_yawns  = 0
#                 stats.consecutive_closed = 0

#             else:
#                 if yawn_label == "YAWN":
#                     if stats.consecutive_yawns >= CONFIG["yawn_streak_needed"]:
#                         alert.trigger("YAWNING")
#                         stats.consecutive_yawns = 0

#                 if eye_label == "CLOSED":
#                     if stats.consecutive_closed >= CONFIG["eye_closed_streak"]:
#                         alert.trigger("EYES CLOSED")
#                         stats.consecutive_closed = 0

#             draw_face_box(frame, x1, y1, x2-x1, y2-y1,
#                           yawn_label, eye_label, yawn_conf)

#         draw_hud(frame, stats, alert, yawn_label, yawn_conf,
#                  eye_label, eye_conf, num_faces, fps, is_drowsy)

#         cv2.imshow("Drowsiness Detection", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord("s"):
#             fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             cv2.imwrite(fn, frame)
#             print(f"[INFO] Saved: {fn}")
#         elif key == ord("r"):
#             stats = StatsTracker()
#             alert.reset()
#             print("[INFO] Reset.")

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"\n  Duration   : {stats.duration()}")
#     print(f"  Yawn rate  : {stats.yawn_rate:.1f}%")
#     print(f"  Eye closed : {stats.eye_closed_rate:.1f}%")
#     print(f"  Alerts     : {alert.alert_count}\n")


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--source",     default=0)
#     p.add_argument("--yawn-model", default=CONFIG["yawn_model_path"])
#     p.add_argument("--eye-model",  default=CONFIG["eye_model_path"])
#     p.add_argument("--debug-eye",  action="store_true",
#                    help="Print raw eye probabilities each frame to verify class mapping")
#     return p.parse_args()


# if __name__ == "__main__":
#     args   = parse_args()
#     source = args.source
#     if isinstance(source, str) and source.isdigit():
#         source = int(source)
#     run_detection(source, args.yawn_model, args.eye_model,
#                   debug_eye=args.debug_eye)

"""
Drowsiness Detection 
Face + Eye (PyTorch MobileNetV2) + Yawn (TF RegNetX)

Alert Logic:
  OPEN   + NO_YAWN = NORMAL            (no alert)
  CLOSED + NO_YAWN = DROWSY            (alert after eye_closed_streak frames)
  OPEN   + YAWN    = DROWSY            (alert after yawn_streak_needed frames)
  CLOSED + YAWN    = DROWSY IMMEDIATE  (alert immediately)
  NO FACE          = FACE NOT VISIBLE  (alert after no_face_alert_frames)

    eye-model drowsy_mobilenetv2.pth
    yawn-model yawn_model.keras
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
import argparse
import time
import os
import sys
from collections import deque
from datetime import datetime

try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


EYE_CLOSED_CLASS_INDEX = 0


CONFIG = {
    "yawn_model_path":       "yawn_model.keras",
    "eye_model_path":        "drowsy_mobilenetv2.pth",
    "yawn_img_size":         (224, 224),
    "eye_img_size":          (224, 224),
    "yawn_class_index":      1,
    "yawn_thresh":           0.50,
    "eye_closed_thresh":     0.65,
    "yawn_streak_needed":    3,
    "eye_closed_streak":     15,
    "no_face_alert_frames":  30,
    "alert_cooldown_sec":    8,
    "yawn_smooth_window":    6,
    "eye_smooth_window":     8,
    "face_scale_factor":     1.1,
    "face_min_neighbors":    5,
    "face_min_size":         (80, 80),
    "font":                  cv2.FONT_HERSHEY_SIMPLEX,
    "display_width":         1280,
    "display_height":        720,
    "alert_sound_path":      "alert.wav",
}

COLOR = {
    "green":  (34, 197, 94),
    "red":    (50,  50, 220),
    "yellow": (0,  200, 255),
    "white":  (255, 255, 255),
    "gray":   (120, 120, 120),
    "dark":   (20,  20,  30),
    "cyan":   (255, 200,   0),
    "orange": (0,  140, 255),
    "accent": (0,  165, 255),
}

#  MODEL LOADERS

def load_yawn_model(path):
    if not os.path.exists(path):
        print(f"[WARN] Yawn model not found: {path}")
        return None
    print(f"[INFO] Loading yawn model: {path}")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(path)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"[INFO] Yawn model loaded. Input: {model.input_shape}")
    return model


def load_eye_model(path):
    if not os.path.exists(path):
        print(f"[WARN] Eye model not found: {path}")
        return None
    print(f"[INFO] Loading eye model: {path}")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Eye model loaded.")
    return model


def load_face_detector():
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if det.empty():
        print("[ERROR] Face detector failed.")
        sys.exit(1)
    print("[INFO] Face detector loaded.")
    return det


def load_eye_cascade():
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    if det.empty():
        return None
    print("[INFO] Eye cascade loaded.")
    return det


def load_mouth_cascade():
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml")
    if det.empty():
        return None
    print("[INFO] Mouth cascade loaded.")
    return det


#  DETECTION HELPERS

def detect_faces(gray, detector):
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=CONFIG["face_scale_factor"],
        minNeighbors=CONFIG["face_min_neighbors"],
        minSize=CONFIG["face_min_size"],
    )
    return list(faces) if len(faces) > 0 else []


def detect_eyes(face_gray, face_x, face_y, face_w, face_h, eye_cascade):
    upper = face_gray[:int(face_h * 0.60), :]
    eyes = eye_cascade.detectMultiScale(
        upper, scaleFactor=1.1, minNeighbors=8,
        minSize=(face_w // 6, face_h // 8))
    return [(face_x + ex, face_y + ey, ew, eh)
            for (ex, ey, ew, eh) in eyes[:2]]


def detect_mouth(face_gray, face_x, face_y, face_w, face_h, mouth_cascade):
    lower_start = int(face_h * 0.45)
    lower = face_gray[lower_start:, :]
    mouths = mouth_cascade.detectMultiScale(
        lower, scaleFactor=1.07, minNeighbors=8,
        minSize=(face_w // 4, face_h // 8))
    if len(mouths) == 0:
        return None
    mouths = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)
    mx, my, mw, mh = mouths[0]
    return (face_x + mx, face_y + lower_start + my, mw, mh)


#  PREPROCESSING

def preprocess_keras(roi):
    """
    Accepts grayscale (H,W) or color (H,W,3) ROI.
    Resizes to yawn_img_size, converts grayscale to 3-channel by stacking,
    normalizes to [0,1], and adds batch dimension.
    """
    img = cv2.resize(roi, CONFIG["yawn_img_size"])
    if len(img.shape) == 2:
        # Grayscale -> stack into 3 identical channels
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(CONFIG["eye_img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_pytorch(roi_bgr):
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    return eye_transform(rgb).unsqueeze(0)


def eye_closed_confidence(model, roi_bgr, debug=False):
    """Returns probability [0..1] that eyes are CLOSED."""
    with torch.no_grad():
        probs = torch.softmax(model(preprocess_pytorch(roi_bgr)), dim=1)[0]
        p0 = float(probs[0])
        p1 = float(probs[1])

    if debug:
        print(f"  [EYE DEBUG] p[0]={p0:.3f}  p[1]={p1:.3f}  "
              f"using index={EYE_CLOSED_CLASS_INDEX}  "
              f"closed_conf={float(probs[EYE_CLOSED_CLASS_INDEX]):.3f}")

    return float(probs[EYE_CLOSED_CLASS_INDEX])


#  ALERT SYSTEM

class AlertSystem:
    def __init__(self):
        self.last_alert_time = 0
        self.alert_active    = False
        self.alert_reason    = ""
        self.alert_count     = 0
        self.sound = None
        if SOUND_AVAILABLE and os.path.exists(CONFIG["alert_sound_path"]):
            try:
                self.sound = pygame.mixer.Sound(CONFIG["alert_sound_path"])
            except Exception:
                pass

    def trigger(self, reason=""):
        now = time.time()
        if now - self.last_alert_time >= CONFIG["alert_cooldown_sec"]:
            self.last_alert_time = now
            self.alert_active    = True
            self.alert_reason    = reason
            self.alert_count    += 1
            print(f"[ALERT] {reason} at {datetime.now().strftime('%H:%M:%S')}")
            if self.sound:
                self.sound.play()

    def reset(self):
        self.alert_active = False


#  STATS TRACKER

class StatsTracker:
    def __init__(self):
        self.total_frames        = 0
        self.yawn_frames         = 0
        self.eye_closed_frames   = 0
        self.consecutive_yawns   = 0
        self.consecutive_closed  = 0
        self.consecutive_no_face = 0
        self.start_time          = time.time()
        self.fps_buffer          = deque(maxlen=30)
        self.yawn_buf            = deque(maxlen=CONFIG["yawn_smooth_window"])
        self.eye_buf             = deque(maxlen=CONFIG["eye_smooth_window"])
        self.prev_time           = time.time()

    def update_fps(self):
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.fps_buffer.append(fps)
        self.prev_time = now
        return float(np.mean(self.fps_buffer))

    def smooth_yawn(self, c):
        self.yawn_buf.append(c)
        return float(np.mean(self.yawn_buf))

    def smooth_eye(self, c):
        self.eye_buf.append(c)
        return float(np.mean(self.eye_buf))

    def duration(self):
        s = int(time.time() - self.start_time)
        return f"{s//60:02d}:{s%60:02d}"

    @property
    def yawn_rate(self):
        return self.yawn_frames / max(self.total_frames, 1) * 100

    @property
    def eye_closed_rate(self):
        return self.eye_closed_frames / max(self.total_frames, 1) * 100


def overlay(img, pt1, pt2, color, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, pt1, pt2, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_bar(frame, x, y, w, h, val, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR["gray"], -1)
    cv2.rectangle(frame, (x, y), (x + int(w * val), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR["white"], 1)
    cv2.putText(frame, f"{label}: {val*100:.0f}%",
                (x, y - 5), CONFIG["font"], 0.4, COLOR["white"], 1, cv2.LINE_AA)


def draw_hud(frame, stats, alert, yawn_label, yawn_conf,
             eye_label, eye_conf, num_faces, fps, is_drowsy):
    h, w = frame.shape[:2]
    f = CONFIG["font"]

    # Top bar
    overlay(frame, (0, 0), (w, 52), COLOR["dark"], 0.8)
    cv2.putText(frame, "DROWSINESS DETECTION SYSTEM",
                (12, 34), f, 0.7, COLOR["accent"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 34), f, 0.6,
                COLOR["green"] if fps >= 15 else COLOR["yellow"], 1, cv2.LINE_AA)
    cv2.putText(frame, stats.duration(), (w - 240, 34), f, 0.6,
                COLOR["white"], 1, cv2.LINE_AA)

    if num_faces == 0:
        status_text  = "NO FACE"
        status_color = COLOR["yellow"]
    else:
        status_text  = "DROWSY" if is_drowsy else "NORMAL"
        status_color = COLOR["red"] if is_drowsy else COLOR["green"]
    overlay(frame, (w - 160, 58), (w - 10, 100), status_color, 0.85)
    cv2.putText(frame, status_text, (w - 150, 90), f, 0.75,
                COLOR["white"], 2, cv2.LINE_AA)

    if alert.alert_active:
        pulse = int(abs(np.sin(time.time() * 4)) * 180)
        overlay(frame, (0, 57), (w - 170, 115), (0, 0, 60 + pulse), 0.88)
        cv2.putText(frame, f"!! ALERT: {alert.alert_reason} !!",
                    (20, 97), f, 0.85, COLOR["white"], 2, cv2.LINE_AA)

    px, py, pw = 10, h - 290, 290
    overlay(frame, (px, py), (px + pw, py + 275), COLOR["dark"], 0.75)

    yc = COLOR["red"] if yawn_label == "YAWN" else COLOR["green"]
    cv2.putText(frame, "YAWN",     (px+10, py+22),  f, 0.45, COLOR["gray"], 1, cv2.LINE_AA)
    cv2.putText(frame, yawn_label, (px+10, py+52),  f, 0.9,  yc, 2, cv2.LINE_AA)
    draw_bar(frame, px+10, py+62, pw-20, 14, yawn_conf, "Conf", COLOR["red"])

    ec = COLOR["red"] if eye_label == "CLOSED" else COLOR["green"]
    cv2.putText(frame, "EYES",    (px+10, py+102), f, 0.45, COLOR["gray"], 1, cv2.LINE_AA)
    cv2.putText(frame, eye_label, (px+10, py+132), f, 0.9,  ec, 2, cv2.LINE_AA)
    draw_bar(frame, px+10, py+142, pw-20, 14, eye_conf, "Conf", COLOR["orange"])

    cv2.putText(frame, f"Faces: {num_faces}   Alerts: {alert.alert_count}",
                (px+10, py+178), f, 0.45, COLOR["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Yawn Rate:  {stats.yawn_rate:.1f}%",
                (px+10, py+198), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eye Closed: {stats.eye_closed_rate:.1f}%",
                (px+10, py+216), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frames: {stats.total_frames}",
                (px+10, py+234), f, 0.43, COLOR["gray"], 1, cv2.LINE_AA)

    # Yawn streak dots
    cv2.putText(frame,
                f"Yawn streak ({stats.consecutive_yawns}/{CONFIG['yawn_streak_needed']})",
                (px+10, py+255), f, 0.4, COLOR["gray"], 1, cv2.LINE_AA)
    for i in range(CONFIG["yawn_streak_needed"]):
        filled = i < stats.consecutive_yawns
        cv2.circle(frame, (px + 15 + i * 22, h - 20), 7,
                   COLOR["red"] if filled else COLOR["gray"], -1 if filled else 2)

    # Eye closed progress bar
    bw    = pw - 20
    ratio = min(stats.consecutive_closed / CONFIG["eye_closed_streak"], 1.0)
    cv2.rectangle(frame, (px+10, h-45), (px+10+bw, h-32), COLOR["gray"], -1)
    cv2.rectangle(frame, (px+10, h-45), (px+10+int(bw*ratio), h-32),
                  COLOR["orange"], -1)
    cv2.putText(frame,
                f"Eye closed: {stats.consecutive_closed}/{CONFIG['eye_closed_streak']}",
                (px+10, h-50), f, 0.38, COLOR["gray"], 1, cv2.LINE_AA)

    # No face progress bar
    if num_faces == 0:
        nf_ratio = min(stats.consecutive_no_face / CONFIG["no_face_alert_frames"], 1.0)
        bx = w - 220
        by = h - 30
        cv2.rectangle(frame, (bx, by), (bx+200, by+12), COLOR["gray"], -1)
        cv2.rectangle(frame, (bx, by), (bx+int(200*nf_ratio), by+12),
                      COLOR["yellow"], -1)
        cv2.putText(frame,
                    f"No face: {stats.consecutive_no_face}/{CONFIG['no_face_alert_frames']}",
                    (bx, by-5), f, 0.4, COLOR["yellow"], 1, cv2.LINE_AA)


def draw_face_box(frame, x, y, w, h, yawn_label, eye_label, yawn_conf):
    is_drowsy = (yawn_label == "YAWN") or (eye_label == "CLOSED")
    color = COLOR["red"] if is_drowsy else COLOR["green"]
    L = 22
    for (px, py), (dx, dy) in [
        ((x, y), (1, 0)), ((x, y), (0, 1)),
        ((x+w, y), (-1, 0)), ((x+w, y), (0, 1)),
        ((x, y+h), (1, 0)), ((x, y+h), (0, -1)),
        ((x+w, y+h), (-1, 0)), ((x+w, y+h), (0, -1))
    ]:
        cv2.line(frame, (px, py), (px + dx*L, py + dy*L), color, 2)

    tag = f"{yawn_label}  EYE:{eye_label}  {yawn_conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(tag, CONFIG["font"], 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(frame, tag, (x + 4, y - 5),
                CONFIG["font"], 0.5, COLOR["white"], 1, cv2.LINE_AA)


#  MAIN LOOP

def run_detection(source, yawn_model_path, eye_model_path, debug_eye=False):
    yawn_model  = load_yawn_model(yawn_model_path)
    eye_model   = load_eye_model(eye_model_path)
    face_det    = load_face_detector()
    eye_cascade = load_eye_cascade()
    mouth_cas   = load_mouth_cascade()
    alert       = AlertSystem()
    stats       = StatsTracker()

    print(f"\n[INFO] Eye closed class index = {EYE_CLOSED_CLASS_INDEX}")
    print(f"[INFO] If eyes are still inverted, change EYE_CLOSED_CLASS_INDEX "
          f"to {1 - EYE_CLOSED_CLASS_INDEX} at the top of this file.\n")
    if debug_eye:
        print("[DEBUG] --debug-eye active: raw probabilities printed each frame.\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["display_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["display_height"])

    print("\n[INFO] Running — Q=quit  S=screenshot  R=reset\n")
    print(f"  Yawn model : {'loaded' if yawn_model else 'MISSING'}")
    print(f"  Eye  model : {'loaded' if eye_model  else 'MISSING'}\n")

    yawn_label = "NO_YAWN"
    eye_label  = "OPEN"
    yawn_conf  = 0.0
    eye_conf   = 0.0
    is_drowsy  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats.total_frames += 1
        fps  = stats.update_fps()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces     = detect_faces(gray, face_det)
        num_faces = len(faces)

        # ── NO FACE DETECTED
        if num_faces == 0:
            stats.consecutive_yawns  = 0
            stats.consecutive_closed = 0
            stats.consecutive_no_face += 1
            is_drowsy  = False
            yawn_label = "NO_YAWN"
            eye_label  = "OPEN"
            yawn_conf  = 0.0
            eye_conf   = 0.0

            if stats.consecutive_no_face >= CONFIG["no_face_alert_frames"]:
                alert.trigger("FACE NOT VISIBLE")
                stats.consecutive_no_face = 0

            cv2.putText(frame, "No face detected",
                        (20, 90), CONFIG["font"], 0.7,
                        COLOR["yellow"], 2, cv2.LINE_AA)

        else:
            stats.consecutive_no_face = 0

            # Pick largest face
            areas = [(fw*fh, (fx, fy, fw, fh)) for (fx, fy, fw, fh) in faces]
            areas.sort(reverse=True)
            fx, fy, fw, fh = areas[0][1]

            # Slightly padded crop
            x1 = max(0, fx - int(fw * 0.08))
            y1 = max(0, fy - int(fh * 0.08))
            x2 = min(frame.shape[1], fx + fw + int(fw * 0.08))
            y2 = min(frame.shape[0], fy + fh + int(fh * 0.08))

            face_bgr  = frame[y1:y2, x1:x2]       # color  — used by eye model
            face_gray = gray[y1:y2, x1:x2]         # gray   — used by yawn model + cascades

            # ── EYE MODEL
            raw_eye    = 0.0
            eye_source = "none"
            face_h_px  = y2 - y1
            face_w_px  = x2 - x1

            if eye_model is not None:
                eyes = detect_eyes(face_gray, x1, y1, face_w_px, face_h_px,
                                   eye_cascade) if eye_cascade else []
                if eyes:
                    eye_source = "cascade"
                    confs = []
                    for (ex, ey, ew, eh) in eyes:
                        roi = frame[ey:ey+eh, ex:ex+ew]
                        if roi.size == 0:
                            continue
                        c = eye_closed_confidence(eye_model, roi, debug=debug_eye)
                        confs.append(c)
                        col = (COLOR["red"] if c > CONFIG["eye_closed_thresh"]
                               else COLOR["green"])
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), col, 1)
                    if confs:
                        raw_eye = max(confs)

                elif face_h_px >= 60 and face_w_px >= 60:
                    eye_source = "geo-fallback"
                    eye_y2 = y1 + int(face_h_px * 0.40)
                    upper_roi = frame[y1:eye_y2, x1:x2]
                    if upper_roi.size > 0:
                        raw_eye = eye_closed_confidence(
                            eye_model, upper_roi, debug=debug_eye)
                    if debug_eye:
                        cv2.rectangle(frame, (x1, y1), (x2, eye_y2),
                                      COLOR["yellow"], 1)
                        cv2.putText(frame, "eye-geo", (x1, y1 - 5),
                                    CONFIG["font"], 0.35, COLOR["yellow"],
                                    1, cv2.LINE_AA)
                else:
                    eye_source = "skip"
                    if debug_eye:
                        print("  [EYE DEBUG] Face ROI too small — skipping eye check")

            if debug_eye and eye_source != "cascade":
                print(f"  [EYE DEBUG] source={eye_source}  raw_eye={raw_eye:.3f}")

            eye_conf  = stats.smooth_eye(raw_eye)
            if eye_source in ("cascade", "geo-fallback"):
                eye_label = "CLOSED" if eye_conf >= CONFIG["eye_closed_thresh"] else "OPEN"
            else:
                eye_label = "OPEN"

            # ── YAWN MODEL (TensorFlow — grayscale input)
            raw_yawn = 0.0
            if yawn_model is not None:
                mouth_input = None

                if mouth_cas is not None:
                    m = detect_mouth(face_gray, x1, y1, x2-x1, y2-y1, mouth_cas)
                    if m is not None:
                        mx, my, mw, mh = m
                        # Pad mouth ROI slightly
                        mx1 = max(0, mx - int(mw * 0.3))
                        my1 = max(0, my - int(mh * 0.3))
                        mx2 = min(frame.shape[1], mx + mw + int(mw * 0.3))
                        my2 = min(frame.shape[0], my + mh + int(mh * 0.3))

                        # ── Grayscale mouth crop
                        mouth_input = gray[my1:my2, mx1:mx2]

                        cv2.rectangle(frame, (mx1, my1), (mx2, my2),
                                      COLOR["cyan"], 2)
                        cv2.putText(frame, "mouth", (mx1, my1 - 5),
                                    CONFIG["font"], 0.38, COLOR["cyan"],
                                    1, cv2.LINE_AA)

                # Fallback: grayscale face crop when no mouth detected
                yawn_input = mouth_input if mouth_input is not None else face_gray

                inp      = preprocess_keras(yawn_input)
                raw_yawn = float(
                    yawn_model.predict(inp, verbose=0)[0][CONFIG["yawn_class_index"]])

            yawn_conf  = stats.smooth_yawn(raw_yawn)
            yawn_label = "YAWN" if yawn_conf >= CONFIG["yawn_thresh"] else "NO_YAWN"

            # ── UPDATE STREAKS
            if yawn_label == "YAWN":
                stats.yawn_frames       += 1
                stats.consecutive_yawns += 1
            else:
                stats.consecutive_yawns = 0

            if eye_label == "CLOSED":
                stats.eye_closed_frames  += 1
                stats.consecutive_closed += 1
            else:
                stats.consecutive_closed = 0

            #   OPEN  + NO_YAWN = NORMAL           → reset
            #   CLOSED + NO_YAWN = DROWSY           → alert after streak
            #   OPEN  + YAWN    = DROWSY            → alert after streak
            #   CLOSED + YAWN   = DROWSY IMMEDIATE  → alert now

            is_drowsy = (eye_label == "CLOSED") or (yawn_label == "YAWN")

            if eye_label == "OPEN" and yawn_label == "NO_YAWN":
                stats.consecutive_yawns  = 0
                stats.consecutive_closed = 0
                alert.reset()

            elif eye_label == "CLOSED" and yawn_label == "YAWN":
                alert.trigger("EYES CLOSED + YAWNING")
                stats.consecutive_yawns  = 0
                stats.consecutive_closed = 0

            else:
                if yawn_label == "YAWN":
                    if stats.consecutive_yawns >= CONFIG["yawn_streak_needed"]:
                        alert.trigger("YAWNING")
                        stats.consecutive_yawns = 0

                if eye_label == "CLOSED":
                    if stats.consecutive_closed >= CONFIG["eye_closed_streak"]:
                        alert.trigger("EYES CLOSED")
                        stats.consecutive_closed = 0

            draw_face_box(frame, x1, y1, x2-x1, y2-y1,
                          yawn_label, eye_label, yawn_conf)

        draw_hud(frame, stats, alert, yawn_label, yawn_conf,
                 eye_label, eye_conf, num_faces, fps, is_drowsy)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[INFO] Saved: {fn}")
        elif key == ord("r"):
            stats = StatsTracker()
            alert.reset()
            print("[INFO] Reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Duration   : {stats.duration()}")
    print(f"  Yawn rate  : {stats.yawn_rate:.1f}%")
    print(f"  Eye closed : {stats.eye_closed_rate:.1f}%")
    print(f"  Alerts     : {alert.alert_count}\n")


#  ENTRY POINT

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",     default=0)
    p.add_argument("--yawn-model", default=CONFIG["yawn_model_path"])
    p.add_argument("--eye-model",  default=CONFIG["eye_model_path"])
    p.add_argument("--debug-eye",  action="store_true",
                   help="Print raw eye probabilities each frame to verify class mapping")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    run_detection(source, args.yawn_model, args.eye_model,
                  debug_eye=args.debug_eye)