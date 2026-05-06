import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
 
# Import your TRUE blueprint
from model_setup import get_model
 
# ========================= CONFIGURATION =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_hybrid_nuclear.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================================
 
def run_explainability(image_path):
    if not os.path.exists(image_path):
        print(f"!!! ERROR: File not found: {image_path}")
        return
 
    print("--- Loading Nuclear Architecture ---")
    model = get_model(MODEL_PATH, device=DEVICE)
 
    # Target the last conv layer for timm EfficientNet
    target_layers = [model.backbone.conv_head]
 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
    img_bgr = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    input_tensor = transform(Image.fromarray(rgb_img_resized)).unsqueeze(0).to(DEVICE)
 
    print(f"--- Generating Forensic Heatmap for {os.path.basename(image_path)} ---")
    targets = [ClassifierOutputTarget(0)]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
 
    img_float = rgb_img_resized / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
 
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {os.path.basename(image_path)}")
    plt.imshow(rgb_img_resized)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap (Detection Focus)")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    save_name = f"heatmap_{os.path.basename(image_path)}"
    save_path = os.path.join(BASE_DIR, "results", save_name)
    plt.savefig(save_path)
    print(f"SUCCESS! Heatmap saved to: {save_path}")
    
    plt.show()
 
if __name__ == "__main__":
    test_img = input("Enter the path of the image to analyze with Grad-CAM: ").strip().replace('"', '')
    
    if not os.path.isabs(test_img) and not os.path.exists(test_img):
        full_path = os.path.join(BASE_DIR, test_img)
    else:
        full_path = test_img
        
    run_explainability(full_path)
plt.show
 









# import torch
# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import models, transforms
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import torch.nn as nn
 
# # ==================== CONFIG ====================
# # DOUBLE CHECK THESE NAMES: Is it .jpg or .jpeg?
# IMAGE_PATH = r"D:\Deepfake_Project\OOD_Test\real_7.jpg"
# MODEL_PATH = r"D:\Deepfake_Project\v2_hybrid_nuclear.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# def run_explainability():
#     # 1. ROBUST PATH CHECKING
#     if not os.path.exists(IMAGE_PATH):
#         print(f"!!! ERROR: File not found: {IMAGE_PATH}")
#         # Let's see what IS in that folder to help you debug
#         folder = os.path.dirname(IMAGE_PATH)
#         if os.path.exists(folder):
#             print(f"Files found in {folder}: {os.listdir(folder)[:5]}...")
#         return
 
#     # 2. LOAD MODEL (Matches your 'Nuclear' setup)
#     print("Loading model...")
#     model = models.efficientnet_b0(weights=None) # Using 'weights' instead of 'pretrained' to avoid warnings
#     num_ftrs = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.5),
#         nn.Linear(num_ftrs, 2)
#     )
#     model.load_state_dict(torch.load(MODEL_PATH))
#     model.to(DEVICE).eval()
 
#     # 3. TARGET THE LAST CONV LAYER
#     target_layers = [model.features[-1]]
 
#     # 4. PREPROCESS IMAGE
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
 
#     img_bgr = cv2.imread(IMAGE_PATH)
#     if img_bgr is None:
#         print(f"!!! ERROR: OpenCV couldn't decode the image at {IMAGE_PATH}")
#         return
 
#     rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     rgb_img_resized = cv2.resize(rgb_img, (224, 224))
#     input_tensor = transform(Image.fromarray(rgb_img_resized)).unsqueeze(0).to(DEVICE)
 
#     # 5. RUN GRAD-CAM
#     print("Generating Heatmap...")
#     cam = GradCAM(model=model, target_layers=target_layers)
    
#     # Target index 0 (Assuming FAKE is class 0)
#     targets = [ClassifierOutputTarget(0)]
 
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
#     # Normalize image for overlay
#     img_float = rgb_img_resized / 255.0
#     visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
 
#     # 6. DISPLAY RESULTS
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title(f"Original: {os.path.basename(IMAGE_PATH)}")
#     plt.imshow(rgb_img_resized)
#     plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.title("Grad-CAM Heatmap (Detection Focus)")
#     plt.imshow(visualization)
#     plt.axis('off')
#     print("Success! Displaying heatmap.")
#     plt.tight_layout()
#     plt.show()
# if __name__ == "__main__":
#     run_explainability()