import os
import warnings
warnings.filterwarnings('ignore')

# Disable watchdog to prevent file watching issues
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="ระบบวิเคราะห์โรคมันสำปะหลัง",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

DISEASE_INFO = {
    'CBB': {
        'name': 'CBB (Cassava Bacterial Blight)',
        'thai_name': 'โรคใบไหม้มันสำปะหลัง',
        'description': 'โรคที่เกิดจากแบคทีเรีย ทำให้ใบมันสำปะหลังเหี่ยวเฉาและตาย',
        'symptoms': ' แสดงอาการใบจุดเหลี่ยมฉํ่านํ้า ใบไหม้ ใบเหี่ยว ยางไหลจนถึงอากยอดเหี่ยวและแห้งตายลงมา นอกจากนี้ยังทำให้ระบบท่อนํ้าอาหารของลำต้นและรากเน่า',
        'treatment': 'ปลูกพืชอายุสั้นเป็นพืชหมุนเวียน, ใช้ท่อนพันธุ์ที่ปราศจากเชื้อ',
        'severity': 'สูง',
        'emoji': '🦠'
    },
    'CBSD': {
        'name': 'CBSD (Cassava Brown Streak Disease)',
        'thai_name': 'โรคลายสีน้ำตาลมันสำปะหลัง',
        'description': 'โรคไวรัสที่ทำให้เกิดลายเส้นสีน้ำตาลบนใบและลำต้น',
        'symptoms': 'ใบเหี่ยวเฉา มีลายสีน้ำตาลบนลำต้น และรากเน่าแห้งแข็ง',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'สูง',
        'emoji': '🍂'
    },
    'CMD': {
        'name': 'CMD (Cassava Mosaic Disease)',
        'thai_name': 'โรคใบด่างมันสำปะหลัง',
        'description': 'ทำให้ต้นมันสำปะหลังมีอาการใบด่างเหลือง ใบเสียรูปทรง หดลดรูป',
        'symptoms': 'ลำต้นแคระแกร็น ไม่เจริญเติบโต หรือมีการเจริญเติบโตน้อย ต้นมันสำปะหลังไม่สร้างหัว',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'ปานกลาง',
        'emoji': '🟡'
    },
    'CGM': {
        'name': 'CGM (Cassava Green Mottle)',
        'thai_name': 'โรคไวรัสใบเขียวมันสำปะหลัง',
        'description': 'จุดสีเขียวหรือจุดสีเหลืองบนใบต้นมันสำปะหลังที่ดูเหมือนโดนวาดไว้',
        'symptoms': 'ใบเกิดการบิดเบี้ยว และเจริญเติบโตช้า ต้นไม้จะตายได้หากโรครุนแรง',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'ต่ำ',
        'emoji': '🟢'
    },
    'HEALTHY': {
        'name': 'Healthy',
        'thai_name': 'สุขภาพดี',
        'description': 'ต้นมันสำปะหลังมีสุขภาพดี ไม่พบโรค',
        'symptoms': 'ใบเขียว สด สุขภาพดี',
        'treatment': 'ดูแลรักษาตามปกติ, ป้องกันโรค',
        'severity': 'ไม่มี',
        'emoji': '✅'
    }
}

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
        b, seq_len, c = x.size()
        y = x.mean(dim=1)
        y = self.fc(y)
        return x * y.unsqueeze(1).expand_as(x)

class CrossStageAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, 64)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x_list):
        B = x_list[0].shape[0]
        x = torch.cat(x_list, dim=1)
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, -1, self.num_heads, t.shape[-1]//self.num_heads).permute(0, 2, 1, 3), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, x.shape[-1])
        
        return self.proj(x.mean(dim=1))

class DynamicFeatureReducer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, 128)
        self.se = SEBlock(128)
        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.se(x)
        return x

class vit_base_patch32_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Using a more widely available model
        self.backbone = timm.create_model(
            'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', 
            pretrained=True,
            num_classes=0
        )
        
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        self.hidden_dim = 768
        self.feature_layers = [9, 11]
        
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
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        intermediate_features = {}
        
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.feature_layers:
                intermediate_features[i] = x
        
        return intermediate_features
    
    def forward(self, x, return_features=False):
        intermediate_features = self.forward_features(x)
        
        reduced_features = []
        for i, layer_idx in enumerate(self.feature_layers):
            feat = intermediate_features[layer_idx]
            reduced = self.reducers[i](feat)
            reduced_features.append(reduced)
        
        x = self.cross_attention(reduced_features)
        
        features = self.classifier[:3](x)
        logits = self.classifier[3:](features)
        logits = logits / self.temperature
        
        if return_features:
            return features
        return logits

def validate_cassava_image(image, model):
    try:
        image = image.resize((448, 448))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_intensity = np.mean(gray_image)
        
        if mean_intensity < 50 or mean_intensity > 200:
            return False, None, None
        
        entropy = -np.sum((gray_image / 255.0) * np.log2(gray_image / 255.0 + 1e-10))
        
        if entropy < 3.0:  
            return False, None, None
        
        return True, None, entropy
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตรวจสอบรูปภาพ: {str(e)}")
        return False, None, None

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device: {device}")
        
        # Print available models for debugging
        st.write("Checking available models...")
        try:
            model_names = timm.list_models('*vit*')
            st.write(f"Available ViT models: {', '.join(model_names[:5])}...")
        except Exception as e:
            st.write(f"Could not list models: {str(e)}")
        
        st.write("Creating model...")
        model = vit_base_patch32_model(num_classes=5)
        model_path = "streamlit/assets/model/best_model.pth"
        
        st.write(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None, None
            
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        st.write("Removing module prefix...")
        checkpoint = remove_module_prefix(checkpoint)

        st.write("Loading state dict...")
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.error(f"Error loading state dict: {str(e)}")
            # Try to print the expected and actual keys
            model_state_dict = model.state_dict()
            st.write("Expected keys:", list(model_state_dict.keys())[:5])
            st.write("Checkpoint keys:", list(checkpoint.keys())[:5])
            return None, None

        model.to(device)
        model.eval()
        st.write("Model loaded successfully!")
        return model, device
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบเส้นทางไฟล์")
        return None, None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        st.write(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        st.write("Traceback:", traceback.format_exc())
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict_disease(model, image, device):
    try:
        image_tensor = preprocess_image(image).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = probabilities.cpu().numpy()[0]
        
        class_names = ['CBB', 'CBSD', 'CMD', 'CGM', 'HEALTHY']
        results = dict(zip(class_names, predictions))
        
        return results
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        return None

def create_prediction_chart(predictions):
    df = pd.DataFrame(list(predictions.items()), columns=['Disease', 'Probability'])
    df['Probability'] = df['Probability'] * 100
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, 
                 x='Probability', 
                 y='Disease',
                 orientation='h',
                 title='ความน่าจะเป็นของแต่ละโรค (%)',
                 color='Probability',
                 color_continuous_scale='RdYlGn_r')
    
    fig.update_layout(
        xaxis_title="ความน่าจะเป็น (%)",
        yaxis_title="ประเภทโรค",
        font=dict(family="Tahoma", size=12),
        height=400
    )
    
    return fig

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.'):
            name = k[7:] 
        new_state_dict[name] = v
    return new_state_dict

def main():
    load_css()
    st.markdown('<div class="farmer-emoji">🌱👨‍🌾🌱</div>', unsafe_allow_html=True)
    st.title("🌿 ระบบวิเคราะห์โรคมันสำปะหลัง")
    st.markdown("### 🔬 AI Image Classification สำหรับเกษตรกรไทย")
    st.markdown("##### 🙅 ไม่แนะนำให้เอาไปใช้กับรูปภาพอื่นนอกจากใบมันสำปะหลัง")

    st.sidebar.markdown("## 📱 การใช้งาน")
    st.sidebar.markdown("""
    1. 📷 ถ่ายรูปหรือเลือกไฟล์รูปภาพ
    2. 🤖 AI จะวิเคราะห์โรค
    3. 📊 ดูผลการวิเคราะห์และคำแนะนำ
    """)

    with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
        model, device = load_model()

    if model is None:
        st.error("❌ ไม่สามารถโหลดโมเดลได้")
        return

    st.markdown("## 📸 เลือกรูปภาพใบมันสำปะหลัง")
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
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🖼️ รูปภาพที่อัพโหลด")
            st.image(image)

        with col2:
            st.markdown("### 🔍 การวิเคราะห์")
            with st.spinner("🤖 AI กำลังวิเคราะห์..."):
                is_valid, confidence_score, entropy = validate_cassava_image(image, model)
                if not is_valid or (confidence_score is not None and confidence_score < 0.5):
                    st.error("❌ ไม่สามารถระบุได้ว่าเป็นรูปใบมันสำปะหลัง หรือความมั่นใจต่ำ")
                    st.warning("""
                        ⚠️ กรุณาตรวจสอบว่า:
                        - รูปภาพเป็นใบมันสำปะหลังจริง
                        - รูปภาพชัดเจน ไม่เบลอ
                        - ถ่ายในระยะใกล้พอที่จะเห็นลักษณะของใบ
                        """)
                    return
                probabilities = predict_disease(model, image, device)
                if probabilities is not None:
                    predicted_class = max(probabilities, key=probabilities.get)
                else:
                    predicted_class = None

            if predicted_class is not None:
                disease_info = DISEASE_INFO[predicted_class]
                confidence = probabilities[predicted_class] * 100
                if predicted_class == 'HEALTHY':
                    card_class = "disease-card healthy-card"
                elif disease_info['severity'] == 'สูง':
                    card_class = "disease-card disease-severe"
                else:
                    card_class = "disease-card"
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>{disease_info['thai_name']}</h3>
                    <p><strong>ชื่อวิทยาศาสตร์:</strong> {disease_info['name']}</p>
                    <p><strong>ความเชื่อมั่น:</strong> {confidence:.1f}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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

        st.markdown("## 📊 ความน่าจะเป็นของแต่ละโรค")
        if probabilities is not None:
            for disease_key, prob in probabilities.items():
                disease_name = DISEASE_INFO[disease_key]['thai_name']
                percentage = float(prob) * 100
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f" {disease_name}")
                    st.progress(float(prob))  
                with col2:
                    st.write(f"{percentage:.1f}%")
    else:
        st.info("📱 กรุณาอัพโหลดรูปภาพใบมันสำปะหลังเพื่อเริ่มการวิเคราะห์")
        st.markdown("## 🌿 ประเภทโรคที่สามารถตรวจจับได้")
        for i, info in DISEASE_INFO.items():
            with st.expander(f"{info['emoji']} {info['thai_name']} ({info['name']})"):
                st.write(f"**คำอธิบาย:** {info['description']}")
                st.write(f"**อาการ:** {info['symptoms']}")
                st.write(f"**การรักษา:** {info['treatment']}")
                if info['severity'] != 'ไม่มี':
                    st.write(f"**ระดับความรุนแรง:** {info['severity']}")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🌾 ระบบวิเคราะห์โรคมันสำปะหลัง <br>
        ⚠️ <em>หมายเหตุ: ผลการวิเคราะห์เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาผู้เชี่ยวชาญเพิ่มเติม</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 