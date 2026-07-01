import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# =====================================================================
# 1. DYNAMIC FILE-PATH RESOLVER & CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_lesion_deploy_model.pth")

CLASS_MAPPING = {
    0: "akiec (Actinic Keratosis / Bowen's Disease)",
    1: "bcc (Basal Cell Carcinoma)",
    2: "bkl (Benign Keratosis-like Lesions)",
    3: "df (Dermatofibroma)",
    4: "mel (Melanoma)",
    5: "nv (Melanocytic Nevi)",
    6: "vasc (Vascular Lesions)"
}

# =====================================================================
# 2. CORE PRODUCTION MODEL ARCHITECTURE DEFINTION
# =====================================================================
@st.cache_resource
def initialize_production_model():
    """Initializes the EfficientNet-B3 network with production parameters"""
    try:
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_MAPPING))
        
        if not os.path.exists(MODEL_PATH):
            return None, f"Missing File Error: '{MODEL_PATH}' was not found."
            
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# =====================================================================
# 3. EXPLAINABLE AI (XAI) GRAD-CAM ENGINE
# =====================================================================
def generate_gradcam(model, input_tensor, target_layer, enhanced_rgb):
    """Generates a visual Grad-CAM heatmap overlay for the skin lesion"""
    # Create placeholders to capture data from hooks
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks onto the final convolutional layer block
    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    # 1. Forward Pass to capture feature map layouts
    model.zero_grad()
    output = model(input_tensor)
    top_class_idx = torch.argmax(output, dim=1).item()
    
    # 2. Backward Pass targeting the specific winning class
    output[0, top_class_idx].backward()

    # Safely disconnect hooks immediately to prevent memory leak crashes
    f_handle.remove()
    b_handle.remove()

    # 3. Extract captured gradients and feature tensors
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = feature_maps[0].cpu().data.numpy()[0]

    # 4. Global Average Pooling over width/height to isolate channel weights
    weights = np.mean(grads, axis=(1, 2))

    # 5. Compute the weighted sum of target activation layers
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * f_maps[i]

    # 6. Apply ReLU filter (keep positive feature attributions only)
    cam = np.maximum(cam, 0)
    
    # Resize map to match image input size and scale boundaries cleanly between 0 and 1
    cam = cv2.resize(cam, (224, 224))
    if np.max(cam) > 0:
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    # 7. Generate color heatmap overlay map and blend with input image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend the original enhanced image and the generated overlay map together (50% each)
    resized_original = cv2.resize(enhanced_rgb, (224, 224))
    blended_output = cv2.addWeighted(resized_original, 0.5, heatmap, 0.5, 0)
    return blended_output

# =====================================================================
# 4. CLINICAL PREPROCESSING ENGINE PIPELINE
# =====================================================================
def process_clinical_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Tier 1: Morphological Blackhat Hair Removal Transform
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_img = cv2.inpaint(img_rgb, thresh, 1, cv2.INPAINT_TELEA)
    
    # Tier 2: LAB-Space CLAHE Contrast Optimization
    lab = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Tier 3: Geometric Tensor Transformations & Normalization
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_img = transform_pipeline(enhanced_img).unsqueeze(0)
    return enhanced_img, tensor_img

# =====================================================================
# 5. STREAMLIT FRONTEND INTERFACE
# =====================================================================
st.set_page_config(page_title="SkinScan AI Portal", page_icon="🩺", layout="wide")

st.title("🩺 Medical Deep Learning Skin Lesion Analyzer with XAI")
st.write("Upload an image to evaluate both classification probability and internal spatial feature attribution loops.")

model, error_msg = initialize_production_model()

if error_msg:
    st.error(f"❌ Error loading model: {error_msg}")
else:
    uploaded_file = st.file_uploader("Choose a dermoscopic file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        with st.spinner("Processing image matrix..."):
            enhanced_preview, input_tensor = process_clinical_image(file_bytes)
            
        # Perform network classification forward pass
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
            
        top_idx = np.argmax(probabilities)
        confidence = probabilities[top_idx] * 100
        predicted_class = CLASS_MAPPING[top_idx]

        # Enable gradients explicitly for the internal Grad-CAM backward calculations
        input_tensor.requires_grad = True
        model.zero_grad()
        
        # TARGET LAYER: Isolate final convolutional map block in the EfficientNet-B3 architecture
        final_conv_layer = model.features[-1]
        
        with st.spinner("Generating Grad-CAM Explainable AI Heatmap..."):
            gradcam_image = generate_gradcam(model, input_tensor, final_conv_layer, enhanced_preview)

        # UI LAYOUT: Display 3 distinct visual stages across a wide multi-column frame
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(file_bytes, caption="1. Uploaded Original Image", use_container_width=True)
        with col2:
            st.image(enhanced_preview, caption="2. Preprocessed (Hair-Removed & CLAHE)", use_container_width=True)
        with col3:
            st.image(gradcam_image, caption="3. Live Grad-CAM XAI Spatial Heatmap", use_container_width=True)
            
        st.success("### 🎉 Diagnostic Classification Report")
        st.metric(label="Predicted Skin Condition Category", value=predicted_class)
        st.metric(label="System Evaluation Confidence Metric", value=f"{confidence:.2f}%")
        
        st.write("#### Probabilities Across All Diagnostic Fields:")
        for idx, name in CLASS_MAPPING.items():
            st.progress(float(probabilities[idx]), text=f"{name}: {probabilities[idx]*100:.2f}%")