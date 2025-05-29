import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import io
import os

def load_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
    }
    
    .stTitle {
        color: #2d5a2d !important;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stHeader {
        color: #3d6b3d !important;
        border-bottom: 2px solid #4a7c4a;
        padding-bottom: 10px;
    }
    
    .disease-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4a7c4a;
        margin: 10px 0;
    }
    
    .healthy-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    
    .disease-severe {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 5px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
        transition: width 0.3s ease;
    }
    
    .farmer-emoji {
        font-size: 2em;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels) for ViT
        b, seq_len, c = x.size()
        # Global average pooling across sequence dimension
        y = x.mean(dim=1)  # (batch, channels)
        y = self.fc(y)     # (batch, channels)
        return x * y.unsqueeze(1).expand_as(x)

# Cross-Stage Attention
class CrossStageAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, 64)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x_list):
        # สำหรับ ViT จะได้ list ของ features จาก intermediate layers
        B = x_list[0].shape[0]
        
        # Concatenate features from different layers
        x = torch.cat(x_list, dim=1)  # (batch, total_seq_len, channels)
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, -1, self.num_heads, t.shape[-1]//self.num_heads).permute(0, 2, 1, 3), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, x.shape[-1])
        
        return self.proj(x.mean(dim=1))

# Dynamic Feature Reducer
class DynamicFeatureReducer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, 128)
        self.se = SEBlock(128)
        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        # x shape: (batch, seq_len, in_channels)
        x = self.proj(x)     # (batch, seq_len, 128)
        x = self.norm(x)
        x = self.se(x)       # Apply SE attention
        return x

# Main Model - ตรงกับโครงสร้างที่คุณมี
class vit_base_patch32_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model(
            'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', 
            pretrained=True,
            num_classes=0  
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Freeze most layers, only train the last few transformer blocks
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in ['blocks.10', 'blocks.11', 'norm', 'head']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # ViT base has 768 hidden dimensions
        self.hidden_dim = 768
        
        # Extract features from intermediate layers (layers 9 and 11)
        self.feature_layers = [9, 11]
        
        # Feature reducers for each selected layer
        self.reducers = nn.ModuleList([
            DynamicFeatureReducer(self.hidden_dim) for _ in self.feature_layers
        ])
        
        self.cross_attention = CrossStageAttention(channels=128)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward_features(self, x):
        # Patch embedding
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        intermediate_features = {}
        
        # Forward through transformer blocks and collect intermediate features
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.feature_layers:
                intermediate_features[i] = x
        
        return intermediate_features
    
    def forward(self, x, return_features=False):
        # Extract intermediate features
        intermediate_features = self.forward_features(x)
        
        # Reduce features from selected layers
        reduced_features = []
        for i, layer_idx in enumerate(self.feature_layers):
            feat = intermediate_features[layer_idx]
            reduced = self.reducers[i](feat)
            reduced_features.append(reduced)
        
        # Apply cross-stage attention
        x = self.cross_attention(reduced_features)
        
        # Classification
        features = self.classifier[:3](x)  # Up to dropout layer
        logits = self.classifier[3:](features)  # Final linear layer
        logits = logits / self.temperature
        
        if return_features:
            return features
        return logits

@st.cache_resource
def load_model():
    try:
        model = vit_base_patch32_model(num_classes=5)
        model = model.to('cpu')  
        
        model_path = os.path.join(os.path.dirname(__file__), 'assets', 'model', 'vit_base_model.pth')
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            st.info(f"Expected model path: {model_path}")
            model = None
            
        return model
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None

# ฟังก์ชันสำหรับ preprocessing รูปภาพ
def preprocess_image(image):
    # Transform สำหรับ ViT patch32 ที่ใช้ input size 448x448
    transform = transforms.Compose([
        transforms.Resize((448, 448)),  # ViT patch32 ใช้ 448x448
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

# ฟังก์ชันสำหรับทำนาย
def predict_disease(model, image):
    if model is None:
        return None, None
    
    try:
        preprocessed = preprocess_image(image)
        
        with torch.no_grad():
            outputs = model(preprocessed)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, probabilities[0].numpy()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        return None, None

# เพิ่มฟังก์ชันใหม่สำหรับตรวจสอบรูปภาพ
def validate_cassava_image(image, model):
    try:
        preprocessed = preprocess_image(image)
        
        with torch.no_grad():
            outputs = model(preprocessed)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob = torch.max(probabilities).item()
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
            
            is_valid = (
                max_prob > 0.7 and  
                entropy < 1.5       
            )
            
            return is_valid, max_prob, entropy
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตรวจสอบรูปภาพ: {str(e)}")
        return False, 0, 0

# ข้อมูลโรคมันสำปะหลัง
DISEASE_INFO = {
    0: {
        'name': 'CBB (Cassava Bacterial Blight)',
        'thai_name': 'โรคใบไหม้แบคทีเรีย',
        'description': 'โรคที่เกิดจากแบคทีเรีย ทำให้ใบมันสำปะหลังเหี่ยวเฉาและตาย',
        'symptoms': 'ใบเหลือง มีจุดน้ำตาล ใบร่วง',
        'treatment': 'ใช้สารป้องกันกำจัดแบคทีเรีย, ตัดส่วนที่เป็นโรคทิ้ง',
        'severity': 'สูง',
        'emoji': '🦠'
    },
    1: {
        'name': 'CBSD (Cassava Brown Streak Disease)',
        'thai_name': 'โรคลายเส้นสีน้ำตาล',
        'description': 'โรคไวรัสที่ทำให้เกิดลายเส้นสีน้ำตาลบนใบและลำต้น',
        'symptoms': 'ลายเส้นสีน้ำตาลบนใบ, หัวเน่า',
        'treatment': 'ใช้พันธุ์ต้านทาน, กำจัดแมลงพาหะ',
        'severity': 'สูง',
        'emoji': '🍂'
    },
    2: {
        'name': 'CMD (Cassava Mosaic Disease)',
        'thai_name': 'โรคใบด่างมันสำปะหลัง',
        'description': 'โรคไวรัสที่ทำให้ใบเกิดลายด่างสีเหลืองและเขียว',
        'symptoms': 'ลายด่างสีเหลือง-เขียวบนใบ, ใบเล็ก บิดเบี้ยว',
        'treatment': 'ใช้พันธุ์ต้านทาน, กำจัดแมลงหวี่ขาว',
        'severity': 'ปานกลาง',
        'emoji': '🟡'
    },
    3: {
        'name': 'CGM (Cassava Green Mottle)',
        'thai_name': 'โรคใบด่างเขียว',
        'description': 'โรคไวรัสที่ทำให้เกิดจุดด่างสีเขียวอ่อนบนใบ',
        'symptoms': 'จุดด่างสีเขียวอ่อน, การเจริญเติบโตช้า',
        'treatment': 'ใช้พันธุ์ต้านทาน, จัดการแมลงพาหะ',
        'severity': 'ต่ำ',
        'emoji': '🟢'
    },
    4: {
        'name': 'Healthy',
        'thai_name': 'สุขภาพดี',
        'description': 'ต้นมันสำปะหลังมีสุขภาพดี ไม่พบโรค',
        'symptoms': 'ใบเขียว สด สุขภาพดี',
        'treatment': 'ดูแลรักษาตามปกติ, ป้องกันโรค',
        'severity': 'ไม่มี',
        'emoji': '✅'
    }
}

def main():
    load_css()
    
    # Header
    st.markdown('<div class="farmer-emoji">🌱👨‍🌾🌱</div>', unsafe_allow_html=True)
    st.title("🌿 ระบบวิเคราะห์โรคมันสำปะหลัง")
    st.markdown("### 🔬 AI Image Classification สำหรับเกษตรกรไทย")
    
    # Sidebar
    st.sidebar.markdown("## 📱 การใช้งาน")
    st.sidebar.markdown("""
    1. 📷 ถ่ายรูปหรือเลือกไฟล์รูปภาพ
    2. 🤖 AI จะวิเคราะห์โรค
    3. 📊 ดูผลการวิเคราะห์และคำแนะนำ
    """)
    with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
        model = load_model()
    
    if model is None:
        st.error("❌ ไม่สามารถโหลดโมเดลได้")
        return
    
    st.markdown("## 📸 เลือกรูปภาพใบมันสำปะหลัง")
    
    # ตัวเลือกการอัพโหลด
    upload_option = st.radio(
        "เลือกวิธีการอัพโหลด:",
        ["📁 เลือกไฟล์จากเครื่อง", "📷 ถ่ายรูปด้วยกล้อง"]
    )
    
    uploaded_file = None
    
    if upload_option == "📁 เลือกไฟล์จากเครื่อง":
        uploaded_file = st.file_uploader(
            "เลือกรูปภาพใบมันสำปะหลัง",
            type=['jpg', 'jpeg', 'png'],
            help="รองรับไฟล์ JPG, JPEG, PNG"
        )
    else:
        camera_image = st.camera_input("ถ่ายรูปใบมันสำปะหลัง")
        if camera_image is not None:
            uploaded_file = camera_image
    
    if uploaded_file is not None:
        # แสดงรูปภาพ
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ รูปภาพที่อัพโหลด")
            st.image(image)
        
        with col2:
            st.markdown("### 🔍 การวิเคราะห์")
            
            with st.spinner("🤖 AI กำลังวิเคราะห์..."):
                # เพิ่มการตรวจสอบรูปภาพ
                is_valid, confidence_score, entropy = validate_cassava_image(image, model)
                
                if not is_valid:
                    st.error("❌ ไม่สามารถระบุได้ว่าเป็นรูปใบมันสำปะหลัง")
                    st.warning("""
                        ⚠️ กรุณาตรวจสอบว่า:
                        - รูปภาพเป็นใบมันสำปะหลังจริง
                        - รูปภาพชัดเจน ไม่เบลอ
                        - ถ่ายในระยะใกล้พอที่จะเห็นลักษณะของใบ
                        """)
                    return
                
                predicted_class, probabilities = predict_disease(model, image)
            
            if predicted_class is not None:
                disease_info = DISEASE_INFO[predicted_class]
                confidence = probabilities[predicted_class] * 100
                
                # การ์ดแสดงผล
                if predicted_class == 4:  # Healthy
                    card_class = "disease-card healthy-card"
                elif disease_info['severity'] == 'สูง':
                    card_class = "disease-card disease-severe"
                else:
                    card_class = "disease-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>{disease_info['emoji']} {disease_info['thai_name']}</h3>
                    <p><strong>ชื่อวิทยาศาสตร์:</strong> {disease_info['name']}</p>
                    <p><strong>ความเชื่อมั่น:</strong> {confidence:.1f}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ข้อมูลรายละเอียด
        st.markdown("## 📋 รายละเอียดการวิเคราะห์")
        
        if predicted_class is not None:
            disease_info = DISEASE_INFO[predicted_class]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📝 คำอธิบาย")
                st.write(disease_info['description'])
                
                st.markdown("### 🔬 อาการ")
                st.write(disease_info['symptoms'])
                
            with col2:
                st.markdown("### 💊 การรักษา/ป้องกัน")
                st.write(disease_info['treatment'])
                
                if disease_info['severity'] != 'ไม่มี':
                    st.markdown(f"### ⚠️ ระดับความรุนแรง: {disease_info['severity']}")
        
        # แสดงความน่าจะเป็นทั้งหมด
        st.markdown("## 📊 ความน่าจะเป็นของแต่ละโรค")
        
        if probabilities is not None:
            for i, prob in enumerate(probabilities):
                disease_name = DISEASE_INFO[i]['thai_name']
                emoji = DISEASE_INFO[i]['emoji']
                percentage = prob * 100
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{emoji} {disease_name}")
                    st.progress(prob)
                with col2:
                    st.write(f"{percentage:.1f}%")
    
    else:
        st.info("📱 กรุณาอัพโหลดรูปภาพใบมันสำปะหลังเพื่อเริ่มการวิเคราะห์")
        
        # แสดงตัวอย่างโรค
        st.markdown("## 🌿 ประเภทโรคที่สามารถตรวจจับได้")
        
        for i, info in DISEASE_INFO.items():
            with st.expander(f"{info['emoji']} {info['thai_name']} ({info['name']})"):
                st.write(f"**คำอธิบาย:** {info['description']}")
                st.write(f"**อาการ:** {info['symptoms']}")
                st.write(f"**การรักษา:** {info['treatment']}")
                if info['severity'] != 'ไม่มี':
                    st.write(f"**ระดับความรุนแรง:** {info['severity']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🌾 ระบบวิเคราะห์โรคมันสำปะหลัง <br>
        ⚠️ <em>หมายเหตุ: ผลการวิเคราะห์เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาผู้เชี่ยวชาญเพิ่มเติม</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()