import streamlit as st
import torch
import cv2
import os
import tempfile
import time
import psutil
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import deque
import mediapipe as mp
from fpdf import FPDF
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import your polished blueprint
from model_setup import get_model
 
# ========================= 1. CYBER-FORENSIC UI SKIN =========================
def apply_cyber_theme():
    st.markdown("""
        <style>
        .main { background-color: #0e1117; color: #00ffcc; font-family: 'Courier New', monospace; }
        .stSidebar { background-color: #161b22; border-right: 1px solid #00ffcc; }
        h1, h2, h3 { color: #00ffcc !important; text-shadow: 0 0 10px #00ffcc; }
        .stMetric { background-color: #1c2128; border: 1px solid #00ffcc; padding: 10px; border-radius: 10px; box-shadow: 0 0 5px #00ffcc; }
        .stButton>button {
            background-color: #00ffcc; color: black; border-radius: 5px;
            font-weight: bold; border: 1px solid #00ffcc; width: 100%;
        }
        .stButton>button:hover { background-color: #161b22; color: #00ffcc; box-shadow: 0 0 15px #00ffcc; }
        
        /* Laser Scanning Animation */
        @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }
        .scanner-line {
            position: absolute; width: 100%; height: 3px;
            background: #00ffcc; box-shadow: 0 0 15px #00ffcc;
            animation: scan 2.5s linear infinite; z-index: 10;
        }
        .video-container { position: relative; border: 2px solid #00ffcc; border-radius: 5px; overflow: hidden; }
        </style>
    """, unsafe_allow_html=True)
 
# ========================= 2. HARDWARE & LOGGING =========================
def log_event(message):
    if "logs" not in st.session_state: st.session_state.logs = []
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(st.session_state.logs) > 8: st.session_state.logs.pop(0)

 
# ========================= 3. PDF REPORT GENERATOR =========================
class ForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'HYBRID NUCLEAR - DEEPFAKE DETECTION LAB', 0, 1, 'L')
        self.set_draw_color(0, 255, 204) 
        self.line(10, 20, 200, 20)
        self.ln(10)

def create_batch_pdf(mode, acc, total, report_dict, name, id, folder_path, cm_path):
    pdf = ForensicReport()
    pdf.add_page()
    
    # --- 1. Technical Forensic Audit Section ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Technical Forensic Audit', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 11)
    # User Info Row
    pdf.cell(100, 8, f"User Name: {name}", 0, 0)
    pdf.cell(0, 8, f"ID Number: {id}", 0, 1)
    
    # Audit Metadata
    pdf.cell(0, 8, f"Date: {time.strftime('%d-%m-%Y | %H:%M:%S')}", 0, 1)
    # pdf.cell(0, 8, f"Hardware: NVIDIA RTX 3060 (CUDA Enabled)", 0, 1)
    pdf.cell(0, 8, f"Hardware: {Device.type.upper()}", 0, 1)

    # Batch & Folder Name (Just the folder name, no full path)
    pdf.cell(0, 8, f"Batch: {mode} Batch", 0, 1)
    folder_name = os.path.basename(folder_path.rstrip("\\/")) # Gets just "OOD_Test"
    pdf.cell(0, 8, f"Folder Name: {folder_name}", 0, 1)
    pdf.ln(10)
 
    # --- 2. Executive Performance Summary Section ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Performance Summary', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Core Metrics
    pdf.cell(0, 8, f"Total Samples Analyzed: {total}", 0, 1)
    pdf.cell(0, 8, f"Overall System Accuracy: {acc}%", 0, 1)
    
    # The Forensic Trio (Pulled from the classification report)
    # We use .get() to avoid errors if the 'FAKE' key is missing
    fake_metrics = report_dict.get('FAKE', report_dict.get('0', {}))
    p = f"{fake_metrics.get('precision', 0)*100:.1f}%"
    r = f"{fake_metrics.get('recall', 0)*100:.1f}%"
    f1 = f"{fake_metrics.get('f1-score', 0)*100:.1f}%"
    
    pdf.cell(0, 8, f"Precision: {p}", 0, 1)
    pdf.cell(0, 8, f"Recall: {r}", 0, 1)
    pdf.cell(0, 8, f"F1-Score: {f1}", 0, 1)
    pdf.ln(10)
 
    # --- 3. Heatmap ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Confusion Matrix (Heatmap Analysis):', 0, 1)
    pdf.image(cm_path, x=15, w=150)
    
    return pdf.output(dest='S').encode('latin-1')

 
# ========================= 4. CONFIG & ENGINE =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="wide")
apply_cyber_theme()
 
def get_last_conv_layer(net):
    for name, module in reversed(list(net.named_modules())):
        if isinstance(module, torch.nn.Conv2d): return module
    return None


@st.cache_resource
def load_forensic_engine():
    return get_model(MODEL_PATH, device=DEVICE)
 
model = load_forensic_engine()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# --- SIDEBAR ---
st.sidebar.title("🕵️ Forensic Dashboard")
# st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Forensic Tool:",
    ["🎥 Video Scanner", "👯 Dual-Video Comparison", "📸 Image XAI", "📊 Metrics"])
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Report Metadata")
user_name = st.sidebar.text_input("User Name:", value="[Your Name]")
id_no = st.sidebar.text_input("ID Number:", value="[Your ID]")

 
st.sidebar.markdown("---")
st.sidebar.info(f"Engine: Hybrid Nuclear, Hardware: {DEVICE.type.upper()}, Target: 0=FAKE, 1=REAL")



 
# MODE 1: VIDEO SCANNER
# =================================================================
if app_mode == "🎥 Video Scanner":
    st.title("Temporal Video Forensics")
    uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
 
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(uploaded_video)
 
        if st.button("Run Forensic Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            col_m1, col_m2 = st.columns(2)
            metric_score = col_m1.empty()
            metric_verdict = col_m2.empty()
            
            # --- PLOT UPGRADE: Split the area into Video (left) and Chart (right) ---
            col_vid, col_plot = st.columns([2, 1])
            with col_vid:
                video_placeholder = st.empty()
            with col_plot:
                st.write("📈 Probability Trace")
                chart_placeholder = st.empty()
            # ------------------------------------------------------------------------
 
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            prediction_buffer = deque(maxlen=15)
            all_scores = []
 
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"Scanning Frame {frame_idx}/{total_frames}")
                if frame_idx % 2 != 0: continue
 
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face.process(rgb_frame)
 
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x1, y1 = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
                        x2, y2 = min(w, int((bbox.xmin + bbox.width) * w)), min(h, int((bbox.ymin + bbox.height) * h))
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                            input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                output = model(input_tensor)
                                probs = torch.softmax(output, dim=1)
                                fake_score = probs[0][0].item()
                            prediction_buffer.append(fake_score)
                            all_scores.append(fake_score)
                            smooth_score = sum(prediction_buffer) / len(prediction_buffer)
                            
                            label, color = (f"FAKE: {smooth_score*100:.1f}%", (255, 0, 0)) if smooth_score > 0.5 else (f"REAL: {(1-smooth_score)*100:.1f}%", (0, 255, 0))
                            metric_score.metric("Deepfake Probability", f"{smooth_score*100:.1f}%")
                            metric_verdict.metric("Current Verdict", "FAKE" if smooth_score > 0.5 else "REAL")
                            
                            # --- TEXT BOUNDARY FIX: Move text inside box if face is too high ---
                            text_y = y1 - 10 if y1 > 20 else y1 + 25
                            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(rgb_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            # -------------------------------------------------------------------
 
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # --- PLOT UPGRADE: Draw the chart ---
                chart_placeholder.line_chart(all_scores)
                # ------------------------------------
                
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"Scanning Frame {frame_idx}/{total_frames}")
            cap.release()
            st.divider()
            if not all_scores:
                st.warning(" EDGE CASE: No facial landmarks detected in the media provided.")
            else:
                final_avg = sum(all_scores) / len(all_scores) if all_scores else 0.5
                final_verdict = "FAKE" if final_avg > 0.5 else "REAL"
                st.subheader(f"🏁 Final Forensic Verdict: {final_verdict} (Confidence: {max(final_avg, 1-final_avg)*100:.2f}%)")
 
            
 
# =================================================================
# MODE 2: DUAL-VIDEO COMPARISON
# =================================================================
elif app_mode == "👯 Dual-Video Comparison":
    st.title(" Side-by-Side Forensics")
    c1, c2 = st.columns(2)
    v1 = c1.file_uploader("Video A (Suspect)", type=['mp4', 'avi', 'mov'])
    v2 = c2.file_uploader("Video B (Reference)", type=['mp4', 'avi', 'mov'])
    
    if v1 and v2:
        if st.button("RUN COMPARATIVE SCAN", type="primary"):
            # Setup temporary storage
            t1, t2 = tempfile.NamedTemporaryFile(delete=False), tempfile.NamedTemporaryFile(delete=False)
            t1.write(v1.read()); t2.write(v2.read())
            
            cap1, cap2 = cv2.VideoCapture(t1.name), cv2.VideoCapture(t2.name)
            p1, p2 = c1.empty(), c2.empty()
            
            # Placeholders for live verdicts
            v_text1, v_text2 = c1.empty(), c2.empty()
            
            # Mediapipe and Score Tracking
            mp_f = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            scores1, scores2 = [], []
            
            while cap1.isOpened() or cap2.isOpened():
                r1, f1 = cap1.read(); r2, f2 = cap2.read()
                
                # Exit if both videos finish
                if not r1 and not r2: break
                
                # Process Stream A
                if f1 is not None:
                    rgb1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                    res1 = mp_f.process(rgb1)
                    if res1.detections:
                        det = res1.detections[0].location_data.relative_bounding_box
                        h, w, _ = f1.shape
                        x, y, fw, fh = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
                        face = f1[max(0,y):y+fh, max(0,x):x+fw]
                        if face.size > 0:
                            t = preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                s = torch.softmax(model(t), dim=1)[0][0].item()
                                scores1.append(s)
                            label, color = (f"FAKE: {s*100:.1f}%", (255,0,0)) if s > 0.5 else (f"REAL: {(1-s)*100:.1f}%", (0,255,0))
                            cv2.rectangle(rgb1, (x, y), (x+fw, y+fh), color, 4)
                            cv2.putText(rgb1, label, (x, y-10 if y > 20 else y+30), 1, 1.5, color, 2)
                    p1.image(rgb1, use_container_width=True)
 
                # Process Stream B
                if f2 is not None:
                    rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    res2 = mp_f.process(rgb2)
                    if res2.detections:
                        det = res2.detections[0].location_data.relative_bounding_box
                        h, w, _ = f2.shape
                        x, y, fw, fh = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
                        face = f2[max(0,y):y+fh, max(0,x):x+fw]
                        if face.size > 0:
                            t = preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                s = torch.softmax(model(t), dim=1)[0][0].item()
                                scores2.append(s)
                            label, color = (f"FAKE: {s*100:.1f}%", (255,0,0)) if s > 0.5 else (f"REAL: {(1-s)*100:.1f}%", (0,255,0))
                            cv2.rectangle(rgb2, (x, y), (x+fw, y+fh), color, 4)
                            cv2.putText(rgb2, label, (x, y-10 if y > 20 else y+30), 1, 1.5, color, 2)
                    p2.image(rgb2, use_container_width=True)
            
            cap1.release(); cap2.release()
            
            # --- FINAL COMPARATIVE SUMMARY ---
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            if scores1:
                final_s1 = sum(scores1)/len(scores1)
                v1 = "FAKE" if final_s1 > 0.5 else "REAL"
                c1 = round(max(final_s1, 1-final_s1)*100, 2)
                col_res1.success(f"Video A Verdict: **{v1}** ({c1}%)")
            
            if scores2:
                final_s2 = sum(scores2)/len(scores2)
                v2 = "FAKE" if final_s2 > 0.5 else "REAL"
                c2 = round(max(final_s2, 1-final_s2)*100, 2)
                col_res2.success(f"Video B Verdict: **{v2}** ({c2}%)")


# =================================================================
# MODE 3: IMAGE SPOT-CHECK (Fixed for EfficientNet)
# =================================================================
elif app_mode == "📸 Image XAI":
    st.title("Spatial Explainability (Grad-CAM)")
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_img:
        col1, col2 = st.columns(2)
        img_pil = Image.open(uploaded_img).convert('RGB')
        with col1:
            st.image(img_pil, caption="Original Image", use_container_width=True)
            
        if st.button("Generate Forensic Heatmap", type="primary"):
            input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                fake_prob = torch.softmax(output, dim=1)[0][0].item()
            
            # DYNAMIC LAYER FIX: Finds the correct layer regardless of architecture
            target_layers = [get_last_conv_layer(model)]
            targets = [ClassifierOutputTarget(0)] 
            
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
            img_resized = np.array(img_pil.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
            
            with col2:
                label = "FAKE" if fake_prob > 0.5 else "REAL"
                st.image(visualization, caption=f"Forensic Focus: {label} ({(fake_prob if label=='FAKE' else 1-fake_prob)*100:.1f}%)", use_container_width=True)

# =================================================================
# MODE 4: SCIENTIFIC METRICS (WITH MANUAL PATH SELECTION)
# =================================================================
elif app_mode == "📊 Metrics":
    st.title("Live System Validation & Benchmarking")
    tab1, tab2 = st.tabs(["🖼️ Image Batch Evaluation", "🎬 Video Batch Evaluation"])
 
    with tab1:
        st.subheader("Spatial Stress Test")
        
        # 1. MANUAL FOLDER SELECTION
        default_img_dir = os.path.join(BASE_DIR, "OOD_Test")
        img_dir = st.text_input("📁 Enter Image Folder Path:", value=default_img_dir, key="img_path_input")
 
        # Check if the path actually exists
        if not os.path.exists(img_dir):
            st.error(f"❌ Folder Not Found! Please check the path: `{img_dir}`")
        else:
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            st.success(f"📂 Linked to: `{os.path.basename(img_dir)}` ({len(img_files)} samples found)")
            
            if st.button("🚀 Run Image Batch Evaluation", key="img_btn"):
                y_true, y_pred = [], []
                p_bar = st.progress(0)
                
                # 2. RUN THE BATCH SCAN
                for i, f in enumerate(img_files):
                    # Labeling: 'fake' in name = 0, else 1
                    actual = 0 if "fake" in f.lower() else 1
                    y_true.append(actual)
                    
                    img_path = os.path.join(img_dir, f)
                    img_p = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        score = torch.softmax(model(img_p), dim=1)[0][0].item()
                    y_pred.append(0 if score > 0.5 else 1)
                    p_bar.progress((i+1)/len(img_files))
 
                # 3. CALCULATE SCIENTIFIC METRICS
                acc_val = accuracy_score(y_true, y_pred) * 100
                report = classification_report(y_true, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
 
                # 4. DISPLAY PERFORMANCE REPORT IN UI
                st.write(f"### Final Image Accuracy: {acc_val:.2f}%")
                
                col_p, col_r, col_f = st.columns(3)
                col_p.metric("Precision", f"{report['FAKE']['precision']*100:.1f}%")
                col_r.metric("Recall (Detection Rate)", f"{report['FAKE']['recall']*100:.1f}%")
                col_f.metric("F1-Score", f"{report['FAKE']['f1-score']*100:.1f}%")
 
                # 5. DRAW THE CONFUSION MATRIX
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='magma', 
                            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax)
                st.pyplot(fig)
 
                # 6. PDF AUDIT GENERATION
                # Ensure results folder exists to save the temporary heatmap
                os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
                cm_p = os.path.join(BASE_DIR, "results", "temp_img_cm.png")
                fig.savefig(cm_p)
                
                # Generate PDF using our updated function
                # Note: user_name and roll_no come from your sidebar inputs
                pdf_data = create_batch_pdf(
                    mode="Image", 
                    acc=f"{acc_val:.2f}", 
                    total=len(img_files), 
                    report_dict=report, 
                    name=user_name, 
                    id=id_no, 
                    folder_path=img_dir, 
                    cm_path=cm_p
                )
                
                st.download_button(
                    label="📥 Download Official Forensic Audit (PDF)", 
                    data=pdf_data, 
                    file_name=f"Forensic_Audit_Image_{time.strftime('%Y%m%d')}.pdf", 
                    mime="application/pdf"
                )
 
        with tab2:
            st.subheader("Temporal Robustness Test")
        
            # 1. MANUAL FOLDER SELECTION
            default_vid_dir = os.path.join(BASE_DIR, "OOD_videos_Test")
            vid_dir = st.text_input("📁 Enter Video Folder Path:", value=default_vid_dir, key="vid_path_input")
 
            # Check if the path actually exists
            if not os.path.exists(vid_dir):
             st.error(f"❌ Folder Not Found! Please check the path: `{vid_dir}`")
            else:
             vid_files = [f for f in os.listdir(vid_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
             st.success(f"📂 Linked to: `{os.path.basename(vid_dir)}` ({len(vid_files)} samples found)")
            
            if st.button("🚀 Run Video Batch Evaluation", key="vid_btn"):
                y_true_v, y_pred_v = [], []
                p_bar_v = st.progress(0)
                mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
                
                # 2. RUN THE BATCH SCAN
                for i, f in enumerate(vid_files):
                    actual = 0 if "fake" in f.lower() else 1
                    y_true_v.append(actual)
                    
                    cap = cv2.VideoCapture(os.path.join(vid_dir, f))
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    step = max(1, total_f // 30)
                    scores, curr = [], 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        if curr % step == 0:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = mp_face.process(rgb)
                            if res.detections:
                                det = res.detections[0].location_data.relative_bounding_box
                                h, w, _ = frame.shape
                                x, y, fw, fh = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
                                face = frame[max(0,y):y+fh, max(0,x):x+fw]
                                if face.size > 0:
                                    tensor = preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                                    with torch.no_grad():
                                        scores.append(torch.softmax(model(tensor), dim=1)[0][0].item())
                        curr += 1
                        if len(scores) >= 35: break
                    
                    cap.release()
                    y_pred_v.append(0 if (sum(scores)/len(scores) if scores else 0.5) > 0.5 else 1)
                    p_bar_v.progress((i+1)/len(vid_files))
 
                # 3. CALCULATE SCIENTIFIC METRICS
                acc_val_v = accuracy_score(y_true_v, y_pred_v) * 100
                report_v = classification_report(y_true_v, y_pred_v, target_names=['FAKE', 'REAL'], output_dict=True)
 
                # 4. DISPLAY PERFORMANCE REPORT IN UI
                st.write(f"### Final Video Accuracy: {acc_val_v:.2f}%")
                
                cp, cr, cf = st.columns(3)
                cp.metric("Precision", f"{report_v['FAKE']['precision']*100:.1f}%")
                cr.metric("Recall (Detection Rate)", f"{report_v['FAKE']['recall']*100:.1f}%")
                cf.metric("F1-Score", f"{report_v['FAKE']['f1-score']*100:.1f}%")
 
                # 5. DRAW THE CONFUSION MATRIX (Using Viridis for Video)
                fig_v, ax_v = plt.subplots()
                sns.heatmap(confusion_matrix(y_true_v, y_pred_v), annot=True, fmt='d', cmap='viridis', 
                            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax_v)
                st.pyplot(fig_v)
 
                # 6. PDF AUDIT GENERATION
                os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
                cm_p_v = os.path.join(BASE_DIR, "results", "temp_vid_cm.png")
                fig_v.savefig(cm_p_v)
                
                # Generate PDF using updated function and variables
                pdf_data_v = create_batch_pdf(
                    mode="Video", 
                    acc=f"{acc_val_v:.2f}", 
                    total=len(vid_files), 
                    report_dict=report_v, 
                    name=user_name, 
                    id=id_no,  
                    folder_path=vid_dir, 
                    cm_path=cm_p_v
                )
                
                st.download_button(
                    label="📥 Download Official Forensic Audit (PDF)", 
                    data=pdf_data_v, 
                    file_name=f"Forensic_Audit_Video_{time.strftime('%Y%m%d')}.pdf", 
                    mime="application/pdf"
                )

















































# import streamlit as st
# import torch
# import cv2
# import os
# import tempfile
# import time
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from collections import deque
# import mediapipe as mp
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
 
# # Import your polished blueprint
# from model_setup import get_model
 
# # ========================= CONFIGURATION =========================
# # Using absolute path logic to prevent "Folder Not Found" issues
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# st.set_page_config(page_title="Deepfake Forensics Lab", page_icon="🕵️", layout="wide")
 
# # Helper to find the last Conv layer automatically (Fixes the AttributeError)
# def get_last_conv_layer(net):
#     for name, module in reversed(list(net.named_modules())):
#         if isinstance(module, torch.nn.Conv2d):
#             return module
#     return None
 
# @st.cache_resource
# def load_forensic_engine():
#     return get_model(MODEL_PATH, device=DEVICE)
 
# model = load_forensic_engine()
 
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# # --- UI SIDEBAR ---
# st.sidebar.title("🕵️ Forensic Dashboard")
# st.sidebar.markdown("---")
# app_mode = st.sidebar.radio("Select Forensic Tool:", 
#                             ["🎥 Video Scanner", "📸 Image Spot-Check (XAI)", "📊 Scientific Metrics"])
# st.sidebar.markdown("---")
# st.sidebar.info(f"Engine: Hybrid Nuclear\nHardware: {DEVICE.type.upper()}\nTarget: 0=FAKE, 1=REAL")
 
# # =================================================================
# # MODE 1: VIDEO SCANNER
# # =================================================================
# if app_mode == "🎥 Video Scanner":
#     st.title("Temporal Video Forensics")
#     uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
    
#     if uploaded_video:
#         tfile = tempfile.NamedTemporaryFile(delete=False) 
#         tfile.write(uploaded_video.read())
#         st.video(uploaded_video)
        
#         if st.button("Run Forensic Analysis", type="primary"):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             col_m1, col_m2 = st.columns(2)
#             metric_score = col_m1.empty()
#             metric_verdict = col_m2.empty()
#             video_placeholder = st.empty()
            
#             cap = cv2.VideoCapture(tfile.name)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
#             prediction_buffer = deque(maxlen=15) 
#             all_scores = [] 
            
#             frame_idx = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret: break
#                 frame_idx += 1
#                 if frame_idx % 2 != 0: continue 
 
#                 h, w, _ = frame.shape
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = mp_face.process(rgb_frame)
 
#                 if results.detections:
#                     for detection in results.detections:
#                         bbox = detection.location_data.relative_bounding_box
#                         x1, y1 = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
#                         x2, y2 = min(w, int((bbox.xmin + bbox.width) * w)), min(h, int((bbox.ymin + bbox.height) * h))
#                         face_img = frame[y1:y2, x1:x2]
#                         if face_img.size > 0:
#                             face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#                             input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
#                             with torch.no_grad():
#                                 output = model(input_tensor)
#                                 probs = torch.softmax(output, dim=1)
#                                 fake_score = probs[0][0].item() 
#                             prediction_buffer.append(fake_score)
#                             all_scores.append(fake_score)
#                             smooth_score = sum(prediction_buffer) / len(prediction_buffer)
#                             label, color = (f"FAKE: {smooth_score*100:.1f}%", (255, 0, 0)) if smooth_score > 0.5 else (f"REAL: {(1-smooth_score)*100:.1f}%", (0, 255, 0))
#                             metric_score.metric("Deepfake Probability", f"{smooth_score*100:.1f}%")
#                             metric_verdict.metric("Current Verdict", "FAKE" if smooth_score > 0.5 else "REAL")
#                             cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
#                             cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
#                 video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
#                 progress_bar.progress(min(frame_idx / total_frames, 1.0))
#                 status_text.text(f"Scanning Frame {frame_idx}/{total_frames}")
#             cap.release()
#             st.divider()
#             final_avg = sum(all_scores) / len(all_scores) if all_scores else 0.5
#             final_verdict = "FAKE" if final_avg > 0.5 else "REAL"
#             st.subheader(f"🏁 Final Forensic Verdict: {final_verdict} (Confidence: {max(final_avg, 1-final_avg)*100:.2f}%)")
 
# # =================================================================
# # MODE 2: IMAGE SPOT-CHECK (Fixed for EfficientNet)
# # =================================================================
# elif app_mode == "📸 Image Spot-Check (XAI)":
#     st.title("Spatial Explainability (Grad-CAM)")
#     uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_img:
#         col1, col2 = st.columns(2)
#         img_pil = Image.open(uploaded_img).convert('RGB')
#         with col1:
#             st.image(img_pil, caption="Original Image", use_container_width=True)
            
#         if st.button("Generate Forensic Heatmap", type="primary"):
#             input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 fake_prob = torch.softmax(output, dim=1)[0][0].item()
            
#             # DYNAMIC LAYER FIX: Finds the correct layer regardless of architecture
#             target_layers = [get_last_conv_layer(model)]
#             targets = [ClassifierOutputTarget(0)] 
            
#             with GradCAM(model=model, target_layers=target_layers) as cam:
#                 grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
#             img_resized = np.array(img_pil.resize((224, 224))) / 255.0
#             visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
            
#             with col2:
#                 label = "FAKE" if fake_prob > 0.5 else "REAL"
#                 st.image(visualization, caption=f"Forensic Focus: {label} ({(fake_prob if label=='FAKE' else 1-fake_prob)*100:.1f}%)", use_container_width=True)
 
# # =================================================================
# # MODE 3: SCIENTIFIC METRICS (Fixed folder detection)
# # =================================================================
# elif app_mode == "📊 Scientific Metrics":
#     st.title("Live System Validation & Benchmarking")
#     tab1, tab2 = st.tabs(["🖼️ Image Batch (OOD_Test)", "🎬 Video Batch (OOD_videos_Test)"])
 
#     with tab1:
#         st.subheader("Spatial Stress Test")
#         IMG_DIR = os.path.join(BASE_DIR, "OOD_Test")
#         # Added debug check
#         if not os.path.exists(IMG_DIR):
#             st.error(f"❌ Folder Not Found! Please ensure images are at: `{IMG_DIR}`")
#         else:
#             img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             st.success(f"📂 Found {len(img_files)} images in `{IMG_DIR}`")
#             if st.button("🚀 Run Image Batch Evaluation", key="img_btn"):
#                 y_true, y_pred = [], []
#                 p_bar = st.progress(0)
#                 for i, f in enumerate(img_files):
#                     actual = 0 if "fake" in f.lower() else 1
#                     y_true.append(actual)
#                     img = preprocess(Image.open(os.path.join(IMG_DIR, f)).convert('RGB')).unsqueeze(0).to(DEVICE)
#                     with torch.no_grad():
#                         score = torch.softmax(model(img), dim=1)[0][0].item()
#                     y_pred.append(0 if score > 0.5 else 1)
#                     p_bar.progress((i+1)/len(img_files))
                
#                 st.write(f"### Final Image Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
#                 fig, ax = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='magma', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax)
#                 st.pyplot(fig)
 
#     with tab2:
#         st.subheader("Temporal Robustness Test")
#         VID_DIR = os.path.join(BASE_DIR, "OOD_videos_Test")
#         # Added debug check
#         if not os.path.exists(VID_DIR):
#             st.error(f"❌ Folder Not Found! Please ensure videos are at: `{VID_DIR}`")
#         else:
#             vid_files = [f for f in os.listdir(VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
#             st.success(f"📂 Found {len(vid_files)} videos in `{VID_DIR}`")
#             if st.button("🚀 Run Video Batch Evaluation", key="vid_btn"):
#                 y_true_v, y_pred_v = [], []
#                 p_bar_v = st.progress(0)
#                 mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
#                 for i, f in enumerate(vid_files):
#                     actual = 0 if "fake" in f.lower() else 1
#                     y_true_v.append(actual)
#                     cap = cv2.VideoCapture(os.path.join(VID_DIR, f))
#                     total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                     step = max(1, total_f // 30)
#                     scores, curr = [], 0
#                     while True:
#                         ret, frame = cap.read()
#                         if not ret: break
#                         if curr % step == 0:
#                             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                             res = mp_face.process(rgb)
#                             if res.detections:
#                                 det = res.detections[0].location_data.relative_bounding_box
#                                 h, w, _ = frame.shape
#                                 x, y, fw, fh = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
#                                 face = frame[max(0,y):y+fh, max(0,x):x+fw]
#                                 if face.size > 0:
#                                     tensor = preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
#                                     with torch.no_grad():
#                                         scores.append(torch.softmax(model(tensor), dim=1)[0][0].item())
#                         curr += 1
#                         if len(scores) >= 35: break
#                     cap.release()
#                     y_pred_v.append(0 if (sum(scores)/len(scores) if scores else 0.5) > 0.5 else 1)
#                     p_bar_v.progress((i+1)/len(vid_files))
                
#                 st.write(f"### Final Video Accuracy: {accuracy_score(y_true_v, y_pred_v)*100:.2f}%")
#                 fig_v, ax_v = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_true_v, y_pred_v), annot=True, fmt='d', cmap='viridis', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax_v)
#                 st.pyplot(fig_v)
 
 


 


















# import streamlit as st
# import torch
# import cv2
# import os
# import tempfile
# import time
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from collections import deque
# import mediapipe as mp
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
 
# # Import your polished blueprint
# from model_setup import get_model
 
# # ========================= CONFIGURATION =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# st.set_page_config(page_title="Deepfake Forensics Lab", page_icon="🕵️", layout="wide")
# # =================================================================
 
# @st.cache_resource
# def load_forensic_engine():
#     return get_model(MODEL_PATH, device=DEVICE)
 
# model = load_forensic_engine()
 
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# # --- UI SIDEBAR ---
# st.sidebar.title("🕵️ Forensic Dashboard")
# st.sidebar.markdown("---")
# app_mode = st.sidebar.radio("Select Forensic Tool:",
#                             ["🎥 Video Scanner", "📸 Image Spot-Check (XAI)", "📊 Scientific Metrics"])
# st.sidebar.markdown("---")
# st.sidebar.info(f"Engine: Hybrid Nuclear\nHardware: {DEVICE.type.upper()}\nTarget: 0=FAKE, 1=REAL")
 
# # =================================================================
# # MODE 1: VIDEO SCANNER
# # =================================================================
# if app_mode == "🎥 Video Scanner":
#     st.title("Temporal Video Forensics")
#     uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
    
#     if uploaded_video:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         st.video(uploaded_video)
        
#         if st.button("Run Forensic Analysis", type="primary"):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             video_placeholder = st.empty()
            
#             cap = cv2.VideoCapture(tfile.name)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
#             prediction_buffer = deque(maxlen=15) # Temporal Consensus Buffer
            
#             frame_idx = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret: break
                
#                 frame_idx += 1
#                 if frame_idx % 2 != 0: continue # Skip frames for UI speed
 
#                 h, w, _ = frame.shape
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = mp_face.process(rgb_frame)
 
#                 if results.detections:
#                     for detection in results.detections:
#                         bbox = detection.location_data.relative_bounding_box
#                         x1, y1 = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
#                         x2, y2 = min(w, int((bbox.xmin + bbox.width) * w)), min(h, int((bbox.ymin + bbox.height) * h))
 
#                         face_img = frame[y1:y2, x1:x2]
#                         if face_img.size > 0:
#                             face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#                             input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
 
#                             with torch.no_grad():
#                                 output = model(input_tensor)
#                                 probs = torch.softmax(output, dim=1)
#                                 fake_score = probs[0][0].item() # 0 is FAKE
                            
#                             prediction_buffer.append(fake_score)
#                             smooth_score = sum(prediction_buffer) / len(prediction_buffer)
 
#                             label, color = (f"FAKE: {smooth_score*100:.1f}%", (255, 0, 0)) if smooth_score > 0.5 else (f"REAL: {(1-smooth_score)*100:.1f}%", (0, 255, 0))
 
#                             cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
#                             cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
#                 video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
#                 progress_bar.progress(min(frame_idx / total_frames, 1.0))
#                 status_text.text(f"Scanning Frame {frame_idx}/{total_frames}")
 
#             cap.release()
#             st.success("Analysis Complete.")
 
# # =================================================================
# # MODE 2: IMAGE SPOT-CHECK (XAI)
# # =================================================================
# elif app_mode == "📸 Image Spot-Check (XAI)":
#     st.title("Spatial Explainability (Grad-CAM)")
#     uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_img:
#         col1, col2 = st.columns(2)
#         img_pil = Image.open(uploaded_img).convert('RGB')
        
#         with col1:
#             st.image(img_pil, caption="Original Image", use_container_width=True)
            
#         if st.button("Generate Forensic Heatmap", type="primary"):
#             input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
            
#             # Predict
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 fake_prob = torch.softmax(output, dim=1)[0][0].item()
            
#             # Grad-CAM logic (Targeting the final convolutional head)
#             target_layers = [model.backbone.conv_head]
#             targets = [ClassifierOutputTarget(0)] # Visualize the "FAKE" class
            
#             with GradCAM(model=model, target_layers=target_layers) as cam:
#                 grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
#             img_resized = np.array(img_pil.resize((224, 224))) / 255.0
#             visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
            
#             with col2:
#                 label = "FAKE" if fake_prob > 0.5 else "REAL"
#                 conf = fake_prob if label == "FAKE" else (1 - fake_prob)
#                 st.image(visualization, caption=f"Forensic Focus: {label} ({conf*100:.1f}%)", use_container_width=True)
 
# # =================================================================
# # MODE 3: SCIENTIFIC METRICS (LIVE DUAL-BATCH)
# # =================================================================
# elif app_mode == "📊 Scientific Metrics":
#     st.title("Live System Validation & Benchmarking")
    
#     tab1, tab2 = st.tabs(["🖼️ Image Batch (OOD_Test)", "🎬 Video Batch (OOD_videos_Test)"])
 
#     # --- IMAGE BATCH ---
#     with tab1:
#         st.subheader("Spatial Stress Test Results")
#         IMG_DIR = os.path.join(BASE_DIR, "OOD_Test")
#         if os.path.exists(IMG_DIR):
#             img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             st.info(f"Loaded {len(img_files)} images from `{IMG_DIR}`")
#             if st.button("🚀 Run Image Batch Evaluation"):
#                 y_true, y_pred = [], []
#                 p_bar = st.progress(0)
#                 for i, f in enumerate(img_files):
#                     actual = 0 if "fake" in f.lower() else 1
#                     y_true.append(actual)
#                     img = preprocess(Image.open(os.path.join(IMG_DIR, f)).convert('RGB')).unsqueeze(0).to(DEVICE)
#                     with torch.no_grad():
#                         score = torch.softmax(model(img), dim=1)[0][0].item()
#                     y_pred.append(0 if score > 0.5 else 1)
#                     p_bar.progress((i+1)/len(img_files))
                
#                 st.write(f"**Final Image Accuracy:** {accuracy_score(y_true, y_pred)*100:.2f}%")
#                 fig, ax = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='magma', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax)
#                 st.pyplot(fig)
 
#     # --- VIDEO BATCH ---
#     with tab2:
#         st.subheader("Temporal Robustness Results")
#         VID_DIR = os.path.join(BASE_DIR, "OOD_videos_Test")
#         if os.path.exists(VID_DIR):
#             vid_files = [f for f in os.listdir(VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
#             st.info(f"Loaded {len(vid_files)} videos from `{VID_DIR}`")
#             if st.button("🚀 Run Video Batch Evaluation"):
#                 y_true_v, y_pred_v = [], []
#                 p_bar_v = st.progress(0)
#                 mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
                
#                 for i, f in enumerate(vid_files):
#                     actual = 0 if "fake" in f.lower() else 1
#                     y_true_v.append(actual)
#                     cap = cv2.VideoCapture(os.path.join(VID_DIR, f))
#                     total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                     step = max(1, total_f // 30)
#                     scores, curr = [], 0
                    
#                     while True:
#                         ret, frame = cap.read()
#                         if not ret: break
#                         if curr % step == 0:
#                             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                             res = mp_face.process(rgb)
#                             if res.detections:
#                                 det = res.detections[0].location_data.relative_bounding_box
#                                 h, w, _ = frame.shape
#                                 x, y, fw, fh = int(det.xmin*w), int(det.ymin*h), int(det.width*w), int(det.height*h)
#                                 face = frame[max(0,y):y+fh, max(0,x):x+fw]
#                                 if face.size > 0:
#                                     tensor = preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
#                                     with torch.no_grad():
#                                         scores.append(torch.softmax(model(tensor), dim=1)[0][0].item())
#                         curr += 1
#                         if len(scores) >= 35: break
#                     cap.release()
#                     y_pred_v.append(0 if (sum(scores)/len(scores) if scores else 0.5) > 0.5 else 1)
#                     p_bar_v.progress((i+1)/len(vid_files))
                
#                 st.write(f"**Final Video Accuracy:** {accuracy_score(y_true_v, y_pred_v)*100:.2f}%")
#                 fig_v, ax_v = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_true_v, y_pred_v), annot=True, fmt='d', cmap='viridis', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax_v)
#                 st.pyplot(fig_v)















# import streamlit as st
# import torch
# import cv2
# import os
# import tempfile
# import time
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from collections import deque
# import mediapipe as mp
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
 
# # Import your polished blueprint
# from model_setup import get_model
 
# # ========================= CONFIGURATION =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# st.set_page_config(page_title="Deepfake Forensics", page_icon="🕵️", layout="wide")
# # =================================================================
 
# @st.cache_resource
# def load_forensic_engine():
#     return get_model(MODEL_PATH, device=DEVICE)
 
# model = load_forensic_engine()
 
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# # --- UI SIDEBAR ---
# st.sidebar.title("🕵️ Forensic Dashboard")
# st.sidebar.markdown("---")
# app_mode = st.sidebar.radio("Select Tool:",
#                             ["🎥 Video Scanner", "📸 Image Spot-Check (XAI)", "📊 Scientific Metrics"])
# st.sidebar.markdown("---")
# st.sidebar.info(f"Engine Status: ONLINE\nHardware: {DEVICE.type.upper()}")
 
# # =================================================================
# # MODE 1: VIDEO SCANNER
# # =================================================================
# if app_mode == "🎥 Video Scanner":
#     st.title("Live Video Forensics")
#     st.markdown("Upload a video to run temporal deepfake detection via the Hybrid Nuclear architecture.")
    
#     uploaded_video = st.file_uploader("Upload MP4 File", type=['mp4', 'mov', 'avi'])
    
#     if uploaded_video is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         st.video(uploaded_video)
        
#         if st.button("Run Forensic Analysis", type="primary"):
#             st.markdown("### Analysis Stream")
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             video_placeholder = st.empty()
            
#             cap = cv2.VideoCapture(tfile.name)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             mp_face_detection = mp.solutions.face_detection
#             face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
#             prediction_buffer = deque(maxlen=15)
            
#             frame_idx = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret: break
                
#                 frame_idx += 1
#                 h, w, _ = frame.shape
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_detector.process(rgb_frame)
 
#                 if results.detections:
#                     for detection in results.detections:
#                         bbox = detection.location_data.relative_bounding_box
#                         x1 = max(0, int(bbox.xmin * w))
#                         y1 = max(0, int(bbox.ymin * h))
#                         x2 = min(w, int((bbox.xmin + bbox.width) * w))
#                         y2 = min(h, int((bbox.ymin + bbox.height) * h))
 
#                         face_img = frame[y1:y2, x1:x2]
#                         if face_img.size > 0:
#                             face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#                             input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
 
#                             with torch.no_grad():
#                                 output = model(input_tensor)
#                                 probs = torch.softmax(output, dim=1)
#                                 fake_score = probs[0][0].item() # Class 0 = FAKE
                            
#                             prediction_buffer.append(fake_score)
#                             smooth_score = sum(prediction_buffer) / len(prediction_buffer)
 
#                             if smooth_score > 0.5:
#                                 label, color = f"FAKE: {smooth_score*100:.1f}%", (255, 0, 0)
#                             else:
#                                 label, color = f"REAL: {(1-smooth_score)*100:.1f}%", (0, 255, 0)
 
#                             cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
#                             cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
#                 video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
#                 progress = min(frame_idx / total_frames, 1.0)
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing Frame {frame_idx}/{total_frames}")
 
#             cap.release()
#             final_avg = sum(prediction_buffer) / len(prediction_buffer) if prediction_buffer else 0
#             verdict = "FAKE" if final_avg > 0.5 else "REAL"
#             confidence = max(final_avg, 1-final_avg) * 100
#             st.success(f"**Video-Level Verdict:** {verdict} ({confidence:.2f}% Confidence)")
 
# # =================================================================
# # MODE 2: IMAGE XAI (Explainable AI)
# # =================================================================
# elif app_mode == "📸 Image Spot-Check (XAI)":
#     st.title("Spatial Explainability (Grad-CAM)")
#     st.markdown("Upload an image to see exactly which facial artifacts triggered the AI's decision.")
    
#     uploaded_img = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_img is not None:
#         col1, col2 = st.columns(2)
#         img_pil = Image.open(uploaded_img).convert('RGB')
#         img_np = np.array(img_pil)
#         img_resized = cv2.resize(img_np, (224, 224))
        
#         with col1:
#             st.image(img_pil, caption="Uploaded Original", use_container_width=True)
            
#         if st.button("Generate Heatmap", type="primary"):
#             input_tensor = preprocess(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
            
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 probs = torch.softmax(output, dim=1)
#                 fake_score = probs[0][0].item()
                
#             label = "FAKE" if fake_score > 0.5 else "REAL"
#             conf = fake_score * 100 if label == "FAKE" else (1 - fake_score) * 100
            
#             target_layers = [model.features[-1]]
#             targets = [ClassifierOutputTarget(0)]
            
#             with GradCAM(model=model, target_layers=target_layers) as cam:
#                 grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
#             img_float = img_resized / 255.0
#             visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            
#             with col2:
#                 st.image(visualization, caption=f"Verdict: {label} ({conf:.2f}%)", use_container_width=True)
 
# # =================================================================
# # MODE 3: SCIENTIFIC METRICS (LIVE)
# # =================================================================
# elif app_mode == "📊 Scientific Metrics":
#     st.title("Live System Validation")
#     TEST_FOLDER = os.path.join(BASE_DIR, "OOD_Test")
    
#     if os.path.exists(TEST_FOLDER):
#         valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#         st.info(f"📁 **Test Directory:** `{TEST_FOLDER}` | **Count:** {len(valid_files)} images.")
    
#         if st.button("🚀 Run Live Batch Evaluation", type="primary"):
#             y_true, y_pred = [], []
#             progress_bar = st.progress(0)
#             start_time = time.time()
            
#             for i, filename in enumerate(valid_files):
#                 actual_is_fake = "fake" in filename.lower()
#                 y_true.append(0 if actual_is_fake else 1)
 
#                 img_path = os.path.join(TEST_FOLDER, filename)
#                 img_tensor = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
                
#                 with torch.no_grad():
#                     output = model(img_tensor)
#                     fake_score = torch.softmax(output, dim=1)[0][0].item()
                
#                 y_pred.append(0 if fake_score > 0.5 else 1)
#                 progress_bar.progress((i + 1) / len(valid_files))
 
#             total_time = time.time() - start_time
#             st.success(f"Evaluation Complete! Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
            
#             fig, ax = plt.subplots()
#             sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='magma',
#                         xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax)
#             st.pyplot(fig)
#     else:
#         st.error("OOD_Test folder not found.")








# import streamlit as st
# import torch
# import cv2
# import os
# import tempfile
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from collections import deque
# import mediapipe as mp
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
 
# from model_setup import get_model
 
# # ========================= CONFIGURATION =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# st.set_page_config(page_title="Deepfake Forensics", page_icon="🕵️", layout="wide")
# # =================================================================
 
# @st.cache_resource
# def load_forensic_engine():
#     return get_model(MODEL_PATH, device=DEVICE)
 
# model = load_forensic_engine()
 
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# # --- UI SIDEBAR ---
# st.sidebar.title("🕵️ Forensic Dashboard")
# st.sidebar.markdown("---")
# app_mode = st.sidebar.radio("Select Tool:",
#                             ["🎥 Video Scanner", "📸 Image Spot-Check (XAI)", "📊 Scientific Metrics"])
# st.sidebar.markdown("---")
# st.sidebar.info(f"Engine Status: ONLINE\nHardware: {DEVICE.type.upper()}")
 
# # =================================================================
# # MODE 1: VIDEO SCANNER
# # =================================================================
# if app_mode == "🎥 Video Scanner":
#     st.title("Live Video Forensics")
#     st.markdown("Upload a video to run temporal deepfake detection via the Hybrid Nuclear architecture.")
    
#     uploaded_video = st.file_uploader("Upload MP4 File", type=['mp4', 'mov', 'avi'])
    
#     if uploaded_video is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         st.video(uploaded_video)
        
#         if st.button("Run Forensic Analysis", type="primary"):
#             st.markdown("### Analysis Stream")
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             video_placeholder = st.empty()
            
#             cap = cv2.VideoCapture(tfile.name)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             mp_face_detection = mp.solutions.face_detection
#             face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
#             prediction_buffer = deque(maxlen=15)
            
#             frame_idx = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret: break
                
#                 frame_idx += 1
#                 h, w, _ = frame.shape
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_detector.process(rgb_frame)
 
#                 if results.detections:
#                     for detection in results.detections:
#                         bbox = detection.location_data.relative_bounding_box
#                         margin_w, margin_h = int((bbox.width * w) * 0.2), int((bbox.height * h) * 0.2)
#                         x1 = max(0, int(bbox.xmin * w) - margin_w)
#                         y1 = max(0, int(bbox.ymin * h) - margin_h)
#                         x2 = min(w, int((bbox.xmin + bbox.width) * w) + margin_w)
#                         y2 = min(h, int((bbox.ymin + bbox.height) * h) + margin_h)
 
#                         face_img = frame[y1:y2, x1:x2]
#                         if face_img.size > 0:
#                             face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#                             input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
 
#                             with torch.no_grad():
#                                 output = model(input_tensor)
#                                 probs = torch.softmax(output, dim=1)
#                                 fake_score = probs[0][0].item() # Class 0 is FAKE
                            
#                             prediction_buffer.append(fake_score)
#                             smooth_score = sum(prediction_buffer) / len(prediction_buffer)
 
#                             if smooth_score > 0.5:
#                                 label, color = f"FAKE: {smooth_score*100:.1f}%", (255, 0, 0)
#                             else:
#                                 label, color = f"REAL: {(1-smooth_score)*100:.1f}%", (0, 255, 0)
 
#                             cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
#                             cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
#                 video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
#                 progress = min(frame_idx / total_frames, 1.0)
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing Frame {frame_idx}/{total_frames}")
 
#             cap.release()
#             final_avg = sum(prediction_buffer) / len(prediction_buffer) if prediction_buffer else 0
#             verdict = "FAKE" if final_avg > 0.5 else "REAL"
#             confidence = max(final_avg, 1-final_avg) * 100
#             st.success(f"**Video-Level Verdict:** {verdict} ({confidence:.2f}% Confidence)")
 
# # =================================================================
# # MODE 2: IMAGE XAI (Explainable AI)
# # =================================================================
# elif app_mode == "📸 Image Spot-Check (XAI)":
#     st.title("Spatial Explainability (Grad-CAM)")
#     st.markdown("Upload an image to see exactly which facial artifacts triggered the AI's decision.")
    
#     uploaded_img = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_img is not None:
#         col1, col2 = st.columns(2)
        
#         img_pil = Image.open(uploaded_img).convert('RGB')
#         img_np = np.array(img_pil)
#         img_resized = cv2.resize(img_np, (224, 224))
        
#         with col1:
#             st.image(img_pil, caption="Uploaded Original", use_container_width=True)
            
#         if st.button("Generate Heatmap", type="primary"):
#             input_tensor = preprocess(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
            
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 probs = torch.softmax(output, dim=1)
#                 fake_score = probs[0][0].item() # Class 0 is FAKE
                
#             label = "FAKE" if fake_score > 0.5 else "REAL"
#             conf = fake_score * 100 if label == "FAKE" else (1 - fake_score) * 100
            
#             # Correct target layer for torchvision EfficientNet
#             target_layers = [model.features[-1]]
#             targets = [ClassifierOutputTarget(0)]
            
#             with GradCAM(model=model, target_layers=target_layers) as cam:
#                 grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
#             img_float = img_resized / 255.0
#             visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            
#             with col2:
#                 st.image(visualization, caption=f"Verdict: {label} ({conf:.2f}%)", use_container_width=True)
#                 if label == "FAKE":
#                     st.error(f"System detected manipulation artifacts. Confidence: {conf:.2f}%")
#                 else:
#                     st.success(f"System verified spatial consistency. Confidence: {conf:.2f}%")
 
# # =================================================================
# # MODE 3: SCIENTIFIC METRICS
# # =================================================================
# elif app_mode == "📊 Scientific Metrics":
#     st.title("Project Evaluation & Metrics")
#     st.markdown("Performance verification of the Hybrid Nuclear architecture.")
    
#     matrix_path = os.path.join(BASE_DIR, "results", "final_benchmark_matrix.png")
    
#     if os.path.exists(matrix_path):
#         st.image(matrix_path, caption="Confusion Matrix: Internal Test Set", width=600)
#     else:
#         st.warning("⚠️ Confusion Matrix not found. Run `test_batch_nuclear.py` in your terminal to generate it.")
        
#     st.markdown("### System Architecture Details")
#     st.write("- **Backbone:** torchvision EfficientNet-B0")
#     st.write("- **Training Strategy:** Hybrid (50% Celeb-DF, 50% Modern OOD)")
#     st.write("- **Loss Function:** CrossEntropy (2-Class)")
#     st.write("- **Face Extraction Pipeline:** MediaPipe BlazeFace (20% Spatial Margin)")