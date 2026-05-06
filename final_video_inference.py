import torch
import cv2
import time
import os
import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms
import mediapipe as mp
 
# 1. Import your polished blueprints
from model_setup import get_model
 
# ========================= CONFIGURATION =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CHANGE THIS to your actual demo video path
INPUT_VIDEO = os.path.join(BASE_DIR, "demo_videos", "test_sample.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "final_demo_labeled.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTHING_WINDOW = 15  # Buffer size for stable labels (0.5 seconds at 30fps)
# =================================================================
 
# 2. INITIALIZE LOGIC
print(f"--- INITIALIZING FORENSIC ENGINE ON: {DEVICE} ---")
model = get_model(MODEL_PATH, device=DEVICE)
 
# MediaPipe Face Detection (Faster and more accurate than Haar Cascades)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
 
# Preprocessing to match the 'Nuclear' training pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# 3. TEMPORAL SMOOTHING TOOLS
# This prevents the 'Real/Fake' box from flickering every frame
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
 
# 4. VIDEO PROCESSING SETUP
cap = cv2.VideoCapture(INPUT_VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))
 
print(f"Video Resolution: {width}x{height} | Target FPS: {fps_in:.1f}")
 
# 5. INFERENCE LOOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
 
    start_time = time.time()
    h, w, _ = frame.shape
    
    # MediaPipe detection requires RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
 
    if results.detections:
        for detection in results.detections:
            # Get Bounding Box
            bbox = detection.location_data.relative_bounding_box
            
            # Add 20% margin to match our extraction polish
            margin_w = int((bbox.width * w) * 0.2)
            margin_h = int((bbox.height * h) * 0.2)
            
            x1 = max(0, int(bbox.xmin * w) - margin_w)
            y1 = max(0, int(bbox.ymin * h) - margin_h)
            x2 = min(w, int((bbox.xmin + bbox.width) * w) + margin_w)
            y2 = min(h, int((bbox.ymin + bbox.height) * h) + margin_h)
 
            # Crop and Predict
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
 
                with torch.no_grad():
                    # Our 'Nuclear' model uses Sigmoid (0 to 1)
                    fake_score = model(input_tensor).item()
                
                # Update Temporal Buffer
                prediction_buffer.append(fake_score)
                # Smoothed Score = Average of the last 15 frames
                smooth_score = sum(prediction_buffer) / len(prediction_buffer)
 
                # Set Label and Color
                if smooth_score > 0.5:
                    label = "FAKE"
                    confidence = smooth_score * 100
                    color = (0, 0, 255) # Red for Fake (BGR)
                else:
                    label = "REAL"
                    confidence = (1 - smooth_score) * 100
                    color = (0, 255, 0) # Green for Real (BGR)
 
                # Draw Visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_text = f"{label}: {confidence:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
 
    # UI OVERLAY (The 'Technical Flex')
    inference_fps = 1.0 / (time.time() - start_time)
    
    # Background for text to make it readable
    cv2.rectangle(frame, (20, 20), (450, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"LOCAL INFERENCE: {inference_fps:.1f} FPS", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"DEVICE: {DEVICE} (RTX 3060)", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
    out.write(frame)
    cv2.imshow("Deepfake Detection System - Forensic UI", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
 
# 6. FINAL SUMMARY
final_avg = sum(prediction_buffer) / len(prediction_buffer) if prediction_buffer else 0
verdict = "FAKE" if final_avg > 0.5 else "REAL"
 
print("\n" + "="*40)
print(f"VIDEO-LEVEL VERDICT: {verdict}")
print(f"FINAL CONFIDENCE: {max(final_avg, 1-final_avg)*100:.2f}%")
print(f"Labeled video saved at: {OUTPUT_PATH}")
print("="*40)
 
cap.release()
out.release()
cv2.destroyAllWindows()









# import torch
# import cv2
# import time
# import torch.nn as nn
# from PIL import Image
# from torchvision import models, transforms
 
# # ==================== CONFIG ====================
# INPUT_VIDEO = r"D:\Deepfake_Project\demo_videos\3_Stress_Tests\stress_fake.mp4" # Path to your demo video
# OUTPUT_PATH = r"D:\Deepfake_Project\results\videos\3_stress_fake_result_demo.mp4"
# MODEL_PATH = r"D:\Deepfake_Project\v2_hybrid_nuclear.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# # 1. LOAD THE MODEL
# model = models.efficientnet_b0(weights=None)
# num_ftrs = model.classifier[1].in_features
# model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))
# model.load_state_dict(torch.load(MODEL_PATH))
# model.to(DEVICE).eval()
 
# # 2. IMAGE PREPROCESSING
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# # 3. FACE DETECTION (OpenCV)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# # 4. VIDEO PROCESSING
# cap = cv2.VideoCapture(INPUT_VIDEO)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps_in = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))
 
# print(f"Running Local Inference on: {DEVICE}")
 
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
 
#     start_time = time.time()
    
#     # Convert to grayscale for detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
 
#     for (x, y, w, h) in faces:
#         # Extract face and predict
#         face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
#         input_tensor = preprocess(Image.fromarray(face_img)).unsqueeze(0).to(DEVICE)
        
#         with torch.no_grad():
#             output = model(input_tensor)
#             prob = torch.softmax(output, dim=1)
#             # Assuming Class 0 is FAKE based on your project history
#             fake_score = prob[0][0].item()
            
#         label = "FAKE" if fake_score > 0.5 else "REAL"
#         confidence = fake_score if label == "FAKE" else (1 - fake_score)
#         color = (0, 0, 255) if label == "FAKE" else (0, 255, 0) # BGR
 
#         # DRAW LABELS (Phase 4 requirement)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
#         cv2.putText(frame, f"{label}: {confidence*100:.1f}%", (x, y-15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
#     # CALCULATE AND DRAW LOCAL FPS
#     end_time = time.time()
#     inference_fps = 1 / (end_time - start_time)
#     cv2.putText(frame, f"LOCAL INFERENCE: {inference_fps:.1f} FPS", (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
 
#     out.write(frame)
#     # Optional: Display in window while processing
#     # cv2.imshow('Deepfake Detection System', frame)
#     # if cv2.waitKey(1) & 0xFF == ord('q'): break
 
# cap.release()
# out.release()
# print(f"Processing complete. Labeled video saved at: {OUTPUT_PATH}")