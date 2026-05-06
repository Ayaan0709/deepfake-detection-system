import torch
import torch.nn as nn
from torchvision import models
 
def get_model(model_path=None, device="cpu"):
    # 1. Load the exact torchvision architecture
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    
    # 2. Rebuild the 2-Class Classifier (0 = Fake, 1 = Real)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    # 3. Load the weights
    if model_path:
        # model.load_state_dict(torch.load(model_path, map_location=device))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    model.to(device)
    model.eval()
    return model











# import torch
# import torch.nn as nn
# import timm
 
# class DeepfakeDetector(nn.Module):
#     """
#     Polished Deepfake Detector Architecture
#     Backbone: EfficientNet-B0 (Optimized for Local Inference)
#     Head: Custom Binary Classifier with Dropout for Robustness
#     """
#     def __init__(self, model_name='efficientnet_b0', pretrained=True):
#         super(DeepfakeDetector, self).__init__()
        
#         # 1. Load the backbone (Features extractor)
#         # We use num_classes=0 to remove the original ImageNet head
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
#         num_features = self.backbone.num_features
 
#         # 2. The 'Head' (Decision Maker)
#         # Dropout(0.6) is key to preventing the model from 'memorizing' specific faces
#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )
 
#     def forward(self, x):
#         features = self.backbone(x)
#         return self.classifier(features)
 
# def get_model(model_path=None, device='cuda'):
#     """
#     Helper function to initialize the model and load weights if provided.
#     """
#     # Auto-detect device if not specified
#     if device == 'cuda' and not torch.cuda.is_available():
#         device = 'cpu'
        
#     model = DeepfakeDetector(pretrained=True if model_path is None else False)
    
#     if model_path:
#         # Load weights and ensure they map to the correct device
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"--- Weights Loaded Successfully: {model_path} ---")
    
#     model.to(device)
#     model.eval() # Default to evaluation mode
#     return model
 
# def count_parameters(model):
#     """A technical flex for your report: Shows model complexity."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
# if __name__ == "__main__":
#     # Test the setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(device=device)
#     print(f"Model initialized on: {device}")
#     print(f"Total Trainable Parameters: {count_parameters(model):,}")





# import torch
# import torch.nn as nn
# import timm

# class DeepfakeDetector(nn.Module):
#     def __init__(self, model_name='efficientnet_b0', pretrained=True):
#         super(DeepfakeDetector, self).__init__()
#         # Load backbone: pretrained=True uses 'Transfer Learning'
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
#         num_features = self.backbone.num_features
        
#         # The 'Head' decides: Real or Fake?
#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(512, 1),
#             nn.Sigmoid() 
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         return self.classifier(x)

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DeepfakeDetector().to(device)
#     print(f"Model initialized on: {device}")