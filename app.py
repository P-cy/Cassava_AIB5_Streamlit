import os
import warnings
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image, ExifTags
import cv2
from torchvision import transforms
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import glob
import requests 
from datetime import datetime, timedelta 
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# --- Weather API Configuration ---
WEATHER_API_KEY = "94f7680251424030b49141625250307" 
WEATHER_API_URL = "http://api.weatherapi.com/v1/history.json"
WEATHER_CURRENT_API_URL = "http://api.weatherapi.com/v1/current.json"  

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
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        padding: 20px;
    }
    
    .stTitle {
        color: #1a472a !important;
        text-align: center;
        font-family: 'Kanit', sans-serif;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    .stMarkdown {
        font-family: 'Kanit', sans-serif;
    }
    
    .disease-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #4a7c4a;
        margin: 15px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.15);
    }
    
    .healthy-card {
        background: linear-gradient(135deg, #e6f5e6 0%, #d4ebd4 100%);
        border-left: 5px solid #28a745;
    }
    
    .disease-severe {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
        border-left: 5px solid #dc3545;
    }
    
    .confidence-bar {
        background: #f8f9fa;
        border-radius: 10px;
        overflow: hidden;
        height: 24px;
        margin: 10px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
        transition: width 0.5s ease;
        border-radius: 10px;
    }
    
    .farmer-emoji {
        font-size: 3em;
        text-align: center;
        margin: 30px 0;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .stButton > button {
        background-color: #1a472a;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2c6e44;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #1a472a;
        box-shadow: 0 0 0 2px rgba(26, 71, 42, 0.2);
    }

    .info-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    .metric-container {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-container:hover {
        transform: translateY(-3px);
    }

    .metric-label {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.4em;
        font-weight: bold;
        color: #1a472a;
    }

    .weather-icon {
        font-size: 2em;
        margin-bottom: 10px;
    }

    .status-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }

    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }

    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .stRadio > div {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .stRadio > div > div > label {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px;
        transition: all 0.3s ease;
    }

    .stRadio > div > div > label:hover {
        background-color: #e9ecef;
    }

    div[data-testid="stFileUploader"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    div[data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    </style>
    
    <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

DISEASE_INFO = {
    'CBB': {
        'name': 'CBB (Cassava Bacterial Blight)',
        'thai_name': 'โรคใบไหม้มันสำปะหลัง',
        'description': 'โรคที่เกิดจากแบคทีเรีย ทำให้ใบมันสำปะหลังเหี่ยวเฉาและตาย',
        'symptoms': ' แสดงอาการใบจุดเหลี่ยมฉ่ำน้ำ ใบไหม้ ใบเหี่ยว ยางไหลจนถึงอากยอดเหี่ยวและแห้งตายลงมา นอกจากนี้ยังทำให้ระบบท่อน้ำอาหารของลำต้นและรากเน่า',
        'treatment': 'ปลูกพืชอายุสั้นเป็นพืชหมุนเวียน, ใช้ท่อนพันธุ์ที่ปราศจากเชื้อ',
        'severity': 'สูง',
        'emoji': '🟡',
        'example_images': sorted(glob.glob('streamlit/assets/img/CBB/*.jpg') + glob.glob('streamlit/assets/img/CBB/*.png'))
    },
    'CBSD': {
        'name': 'CBSD (Cassava Brown Streak Disease)',
        'thai_name': 'โรคลายสีน้ำตาลมันสำปะหลัง',
        'description': 'โรคไวรัสที่ทำให้เกิดลายเส้นสีน้ำตาลบนใบและลำต้น',
        'symptoms': 'ใบเหี่ยวเฉา มีลายสีน้ำตาลบนลำต้น และรากเน่าแห้งแข็ง',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'สูง',
        'emoji': '🍂',
        'example_images': sorted(glob.glob('streamlit/assets/img/CBSD/*.jpg') + glob.glob('streamlit/assets/img/CBSD/*.png'))
    },
    'CGM': {
        'name': 'CGM (Cassava Green Mottle)',
        'thai_name': 'โรคไวรัสใบเขียวมันสำปะหลัง',
        'description': 'จุดสีเขียวหรือจุดสีเหลืองบนใบต้นมันสำปะหลังที่ดูเหมือนโดนวาดไว้',
        'symptoms': 'ลำต้นแคระแกร็น ไม่เจริญเติบโต หรือมีการเจริญเติบโตน้อย ต้นมันสำปะหลังไม่สร้างหัว',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'ต่ำ',
        'emoji': '🦠',
        'example_images': sorted(glob.glob('streamlit/assets/img/CGM/*.jpg') + glob.glob('streamlit/assets/img/CGM/*.png'))
    },
    'CMD': {
        'name': 'CMD (Cassava Mosaic Disease)',
        'thai_name': 'โรคใบด่างมันสำปะหลัง',
        'description': 'ทำให้ต้นมันสำปะหลังมีอาการใบด่างเหลือง ใบเสียรูปทรง หดลดรูป',
        'symptoms': 'ใบเกิดการบิดเบี้ยว และเจริญเติบโตช้า ต้นไม้จะตายได้หากโรครุนแรง',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'ปานกลาง',
        'emoji': '🟢',
        'example_images': sorted(glob.glob('streamlit/assets/img/CMD/*.jpg') + glob.glob('streamlit/assets/img/CMD/*.png'))
    },
    'HEALTHY': {
        'name': 'Healthy',
        'thai_name': 'สุขภาพดี',
        'description': 'ต้นมันสำปะหลังมีสุขภาพดี ไม่พบโรค',
        'symptoms': 'ใบเขียว สด สุขภาพดี',
        'treatment': 'ดูแลรักษาตามปกติ, ป้องกันโรค',
        'severity': 'ไม่มี',
        'emoji': '✅',
        'example_images': sorted(glob.glob('streamlit/assets/img/HEALTY/*.jpg') + glob.glob('streamlit/assets/img/HEALTY/*.png'))
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

def fix_image_orientation(image):
    """Fix image orientation based on EXIF data"""
    try:
        # Get EXIF data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            
            # Rotate image based on EXIF orientation
            if orientation_value == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 3:
                image = image.rotate(180)
            elif orientation_value == 4:
                image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 5:
                image = image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 6:
                image = image.rotate(-90)
            elif orientation_value == 7:
                image = image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 8:
                image = image.rotate(90)
    except (AttributeError, KeyError, IndexError):
        # Cases when Exif data is not available/valid
        pass
    return image

def validate_cassava_image(image, model):
    try:
        # Fix image orientation first
        image = fix_image_orientation(image)
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
        model = vit_base_patch32_model(num_classes=5)
        model_path = "streamlit/assets/model/best_model.pth"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None, None
            
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        checkpoint = remove_module_prefix(checkpoint)

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
    # Fix image orientation first
    image = fix_image_orientation(image)
    
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
        
        class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'HEALTHY']
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

def get_user_location():
    try:
        response = requests.get('http://ip-api.com/json/')
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data['lat'], data['lon'], data['city'] + ', ' + data['country']
    except:
        pass
    return 13.7367, 100.5532, "Bangkok, Thailand"

def reverse_geocode(lat, lon):
    try:
        geolocator = Nominatim(user_agent="cassava_disease_analyzer")
        location = geolocator.reverse(f"{lat}, {lon}", language='th')
        if location:
            return location.address
        else:
            return f"ละติจูด: {lat:.4f}, ลองจิจูด: {lon:.4f}"
    except:
        return f"ละติจูด: {lat:.4f}, ลองจิจูด: {lon:.4f}"

def location_input_section():
    st.markdown("## 📍 เลือกตำแหน่งที่ถ่ายภาพ")
    
    location_method = st.radio(
        "เลือกวิธีการระบุตำแหน่ง:",
        ["🗺️ เลือกจากแผนที่", "⌨️ กรอกพิกัดเอง", "📍 ใช้ตำแหน่งปัจจุบัน"]
    )
    
    if 'selected_lat' not in st.session_state:
        st.session_state.selected_lat = 16.4883
    if 'selected_lon' not in st.session_state:
        st.session_state.selected_lon = 102.8340
    if 'location_name' not in st.session_state:
        st.session_state.location_name = "ขอนแก่น, ประเทศไทย"
    
    latitude = st.session_state.selected_lat
    longitude = st.session_state.selected_lon
    
    if location_method == "🗺️ เลือกจากแผนที่":
        st.write("👆 คลิกบนแผนที่เพื่อเลือกตำแหน่ง")
        
        m = folium.Map(location=[latitude, longitude], zoom_start=10, tiles='OpenStreetMap')
        folium.Marker(
            [latitude, longitude],
            popup=f"ตำแหน่งที่เลือก<br>ละติจูด: {latitude:.4f}<br>ลองจิจูด: {longitude:.4f}",
            tooltip="ตำแหน่งที่เลือก",
            icon=folium.Icon(color='red', icon='camera')
        ).add_to(m)
        
        map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
        
        if map_data['last_clicked'] is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            st.session_state.selected_lat = clicked_lat
            st.session_state.selected_lon = clicked_lon
            
            with st.spinner("กำลังค้นหาชื่อสถานที่..."):
                location_name = reverse_geocode(clicked_lat, clicked_lon)
                st.session_state.location_name = location_name
            
            st.rerun()
        
        st.success(f"📍 ตำแหน่งที่เลือก: {st.session_state.location_name}")
        st.info(f"ละติจูด: {latitude:.6f}, ลองจิจูด: {longitude:.6f}")
        
    elif location_method == "⌨️ กรอกพิกัดเอง":
        col1, col2 = st.columns(2)
        with col1:
            new_lat = st.number_input("ละติจูด (Latitude)", 
                                    value=float(latitude), 
                                    min_value=-90.0, 
                                    max_value=90.0, 
                                    step=0.000001,
                                    format="%.6f",
                                    help="ตัวอย่าง: 16.488300 (ขอนแก่น)")
        with col2:
            new_lon = st.number_input("ลองจิจูด (Longitude)", 
                                    value=float(longitude), 
                                    min_value=-180.0, 
                                    max_value=180.0, 
                                    step=0.000001,
                                    format="%.6f",
                                    help="ตัวอย่าง: 102.834000 (ขอนแก่น)")
        
        if new_lat != latitude or new_lon != longitude:
            st.session_state.selected_lat = new_lat
            st.session_state.selected_lon = new_lon
            
            with st.spinner("กำลังค้นหาชื่อสถานที่..."):
                location_name = reverse_geocode(new_lat, new_lon)
                st.session_state.location_name = location_name
            
            latitude = new_lat
            longitude = new_lon
        
        if st.button("🔍 ดูตำแหน่งบนแผนที่"):
            m = folium.Map(location=[latitude, longitude], zoom_start=15)
            folium.Marker(
                [latitude, longitude],
                popup=f"ตำแหน่งที่ระบุ<br>{st.session_state.location_name}",
                tooltip="ตำแหน่งที่ระบุ",
                icon=folium.Icon(color='green', icon='check')
            ).add_to(m)
            st_folium(m, width=700, height=300)
        
        st.info(f"📍 ตำแหน่ง: {st.session_state.location_name}")
    
    elif location_method == "📍 ใช้ตำแหน่งปัจจุบัน":
        if st.button("🔍 ตรวจหาตำแหน่งปัจจุบัน"):
            with st.spinner("กำลังค้นหาตำแหน่งปัจจุบัน..."):
                current_lat, current_lon, current_location = get_user_location()
                st.session_state.selected_lat = current_lat
                st.session_state.selected_lon = current_lon
                st.session_state.location_name = current_location
                st.rerun()
        
        st.info(f"📍 ตำแหน่งปัจจุบัน: {st.session_state.location_name}")
        st.info(f"ละติจูด: {latitude:.6f}, ลองจิจูด: {longitude:.6f}")
        
        m = folium.Map(location=[latitude, longitude], zoom_start=12)
        folium.Marker(
            [latitude, longitude],
            popup=f"ตำแหน่งปัจจุบัน<br>{st.session_state.location_name}",
            tooltip="ตำแหน่งปัจจุบัน",
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(m)
        st_folium(m, width=700, height=400)
    
    return latitude, longitude

def display_image_slider(images, caption=""):
    if not images:
        st.warning("ไม่พบรูปภาพตัวอย่าง")
        return
        
    if f"image_index_{caption}" not in st.session_state:
        st.session_state[f"image_index_{caption}"] = 0
    
    current_index = st.session_state[f"image_index_{caption}"]
    
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("⬅️ ก่อนหน้า", key=f"prev_{caption}"):
            st.session_state[f"image_index_{caption}"] = (current_index - 1) % len(images)
            st.rerun()
    
    with col2:
        img = Image.open(images[current_index])
        st.image(img, caption=caption, use_container_width=True)
    
    with col3:
        if st.button("ถัดไป ➡️", key=f"next_{caption}"):
            st.session_state[f"image_index_{caption}"] = (current_index + 1) % len(images)
            st.rerun()
    
    st.write(f"รูปที่ {current_index + 1} จาก {len(images)}")

# --- New Function: Load Weather Rules from CSV ---
@st.cache_data
def load_weather_rules(file_path="weather_rules.csv"):
    try:
        rules_df = pd.read_csv(file_path)
        # Don't modify the column names - keep them as is in Thai
        return rules_df
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์กฎเกณฑ์สภาพอากาศ: {file_path}. กรุณาตรวจสอบว่าคุณได้สร้างไฟล์นี้แล้ว")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์กฎเกณฑ์สภาพอากาศ: {str(e)}")
        return pd.DataFrame()

# --- New Function: Get Historical Weather Data ---
@st.cache_data(ttl=timedelta(hours=1))
def get_historical_weather(latitude, longitude, date_str, days_back=7):
    try:
        # Validate coordinates
        try:
            lat = float(latitude)
            lon = float(longitude)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return None, "ค่าละติจูดหรือลองจิจูดไม่ถูกต้อง (ละติจูด: -90 ถึง 90, ลองจิจูด: -180 ถึง 180)"
        except ValueError:
            return None, "กรุณากรอกค่าละติจูดและลองจิจูดเป็นตัวเลขเท่านั้น"

        # Validate date
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            if target_date > today:
                return None, "ไม่สามารถดึงข้อมูลสภาพอากาศในอนาคตได้"
            if target_date < today - timedelta(days=7):
                return None, "สามารถดึงข้อมูลย้อนหลังได้ไม่เกิน 7 วัน"
        except ValueError:
            return None, "รูปแบบวันที่ไม่ถูกต้อง"

        historical_data = {}
        days_to_fetch = min(days_back, (today - target_date).days + 1)

        for i in range(days_to_fetch):
            current_date = target_date - timedelta(days=i)
            date_param = current_date.strftime("%Y-%m-%d")
            
            # Skip future dates
            if current_date > today:
                continue

            params = {
                "key": WEATHER_API_KEY,
                "q": f"{lat},{lon}",
                "dt": date_param
            }
            
            try:
                response = requests.get(WEATHER_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "forecast" in data and "forecastday" in data["forecast"] and len(data["forecast"]["forecastday"]) > 0:
                    day_data = data["forecast"]["forecastday"][0]["day"]
                    historical_data[date_param] = {
                        "avgtemp_c": day_data.get("avgtemp_c"),
                        "totalprecip_mm": day_data.get("totalprecip_mm"),
                        "avghumidity": day_data.get("avghumidity")
                    }
                else:
                    return None, f"ไม่พบข้อมูลสภาพอากาศสำหรับวันที่ {date_param}"
                    
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if "Bad Request" in error_msg:
                    return None, "พิกัดที่ระบุไม่ถูกต้องหรือไม่พบข้อมูลสภาพอากาศ"
                elif "Unauthorized" in error_msg:
                    return None, "API Key ไม่ถูกต้องหรือหมดอายุ"
                else:
                    return None, f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Weather API: {error_msg}"

        if not historical_data:
            return None, "ไม่พบข้อมูลสภาพอากาศในช่วงวันที่ระบุ"

        return historical_data, None

    except Exception as e:
        return None, f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"

# --- New Function: Get Current Weather Data ---
def get_current_weather(latitude, longitude):
    """ดึงข้อมูลสภาพอากาศปัจจุบัน"""
    try:
        # Validate coordinates
        try:
            lat = float(latitude)
            lon = float(longitude)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return None, "ค่าละติจูดหรือลองจิจูดไม่ถูกต้อง"
        except ValueError:
            return None, "กรุณากรอกค่าละติจูดและลองจิจูดเป็นตัวเลขเท่านั้น"

        params = {
            "key": WEATHER_API_KEY,
            "q": f"{lat},{lon}"
        }

        response = requests.get(WEATHER_CURRENT_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "current" in data:
            current_data = {
                "temp_c": data["current"].get("temp_c"),
                "humidity": data["current"].get("humidity"),
                "precip_mm": data["current"].get("precip_mm"),
                "condition": data["current"].get("condition", {}).get("text"),
                "wind_kph": data["current"].get("wind_kph"),
                "last_updated": data["current"].get("last_updated")
            }
            return current_data, None
        else:
            return None, "ไม่พบข้อมูลสภาพอากาศปัจจุบัน"

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "Bad Request" in error_msg:
            return None, "พิกัดที่ระบุไม่ถูกต้องหรือไม่พบข้อมูลสภาพอากาศ"
        elif "Unauthorized" in error_msg:
            return None, "API Key ไม่ถูกต้องหรือหมดอายุ"
        else:
            return None, f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Weather API: {error_msg}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"

# --- New Function: Evaluate Weather Rules ---
def evaluate_weather_rules(predicted_disease_key, weather_data_summary, rules_df):
    messages = []
    
    # ตรวจสอบว่ามีข้อมูลครบถ้วนหรือไม่
    missing_data = []
    if weather_data_summary.get('avg_temp_c') is None:
        missing_data.append("อุณหภูมิ")
    if weather_data_summary.get('avg_humidity') is None:
        missing_data.append("ความชื้นสัมพัทธ์")
    if weather_data_summary.get('total_precip_mm') is None:
        missing_data.append("ปริมาณน้ำฝน")
    
    if missing_data:
        messages.append(f"⚠️ ไม่สามารถประเมินผลจาก {', '.join(missing_data)} ได้ เนื่องจากไม่มีข้อมูล")
        return messages
    
    # Filter rules for the predicted disease
    relevant_rules = rules_df[rules_df['โรค'].astype(str) == predicted_disease_key.upper()]
    
    
    # แสดงผลการวิเคราะห์
    messages.append("### 🔍 ผลการวิเคราะห์สภาพแวดล้อม")
    
    for index, row in relevant_rules.iterrows():
        param = str(row['พารามิเตอร์สภาพอากาศหลัก']).strip()
        condition_str = str(row['ช่วงที่เหมาะสม']).strip()
        note = str(row['เงื่อนไขเฉพาะ']).strip()

        is_match = False
        current_param_value = None

        # Map parameters to their values
        param_mapping = {
            'ความชื้นสัมพัทธ์': 'avg_humidity',
            'อุณหภูมิ': 'avg_temp_c',
            'ปริมาณน้ำฝน': 'total_precip_mm',
            'ความชื้น': 'avg_humidity',
            'สภาพอากาศ': 'total_precip_mm'
        }

        if param in param_mapping:
            current_param_value = weather_data_summary.get(param_mapping[param])
            if current_param_value is None:
                continue

        try:
            clean_condition = condition_str.replace('°C', '').replace('%', '').replace('มม.', '').replace('มม./วัน', '').strip()
            
            if '->' in clean_condition:
                low, high = map(float, clean_condition.split('->'))
                is_match = low <= current_param_value <= high
                condition_display = f"{low}°C ถึง {high}°C" if param == 'อุณหภูมิ' else f"{low}% ถึง {high}%" if param == 'ความชื้นสัมพัทธ์' else f"{low} ถึง {high} มม."
            
            elif '>=' in clean_condition:
                value = float(clean_condition.replace('>=', '').strip())
                is_match = current_param_value >= value
                condition_display = f"มากกว่าหรือเท่ากับ {value}" + ("°C" if param == 'อุณหภูมิ' else "%" if param == 'ความชื้นสัมพัทธ์' else " มม.")
            
            elif '<=' in clean_condition:
                value = float(clean_condition.replace('<=', '').strip())
                is_match = current_param_value <= value
                condition_display = f"น้อยกว่าหรือเท่ากับ {value}" + ("°C" if param == 'อุณหภูมิ' else "%" if param == 'ความชื้นสัมพัทธ์' else " มม.")
            
            elif condition_str == 'เหมาะสม':
                if param == 'อุณหภูมิ':
                    is_match = 25 <= current_param_value <= 32
                    condition_display = "25°C ถึง 32°C"
                elif param == 'ความชื้นสัมพัทธ์':
                    is_match = 70 <= current_param_value <= 80
                    condition_display = "70% ถึง 80%"
                elif param == 'ปริมาณน้ำฝน':
                    is_match = current_param_value >= 35
                    condition_display = "มากกว่า 35 มม."
            
            elif condition_str == 'แห้งแล้ง':
                is_match = current_param_value < 10
                condition_display = "น้อยกว่า 10 มม."
            
            elif condition_str == 'เริ่มต้นฤดูฝน':
                is_match = current_param_value > 20
                condition_display = "มากกว่า 20 มม."

            # แสดงผลการวิเคราะห์
            value_str = f"{current_param_value:.1f}"
            unit = "°C" if param == 'อุณหภูมิ' else "%" if param == 'ความชื้นสัมพัทธ์' else "มม."
            
            if is_match:
                messages.append(f"⚠️ **{param}** ({value_str}{unit}) - {note}")
            else:
                messages.append(f"✅ **{param}** ({value_str}{unit}) - ไม่เอื้อต่อการเกิดโรค")

        except Exception as e:
            continue

    messages.append("---")
    if predicted_disease_key == 'HEALTHY':
        messages.append("### ✅ สรุป: สภาพอากาศเหมาะสมต่อการเจริญเติบโตของมันสำปะหลัง")
    else:
        found_risks = any("⚠️" in msg for msg in messages)
        if found_risks:
            messages.append(f"### ⚠️ สรุป: พบปัจจัยเสี่ยงต่อการเกิดโรค {DISEASE_INFO[predicted_disease_key]['thai_name']}")
        else:
            messages.append(f"### ✅ สรุป: สภาพอากาศโดยรวมไม่เอื้อต่อการเกิดโรค {DISEASE_INFO[predicted_disease_key]['thai_name']}")

    return messages

def get_risk_priority(risk_level):
    """Return priority number for risk level"""
    risk_priorities = {
        'เสี่ยงสูง': 3,
        'เสี่ยงปานกลาง': 2,
        'เสี่ยงต่ำ': 1,
        'สุขภาพดี': 0
    }
    return risk_priorities.get(risk_level, 0)

def main():
    load_css()
    
    st.markdown('<div class="farmer-emoji">🌱👨‍🌾🌱</div>', unsafe_allow_html=True)
    st.title("ระบบวิเคราะห์โรคมันสำปะหลัง")
    st.markdown("### AI Image Classification สำหรับเกษตรกรไทย")

    with st.sidebar:
        st.markdown("## 📱 วิธีการใช้งาน")
        st.markdown("""
        <div class="info-box">
            <ol>
                <li>📸 ถ่ายรูปหรือเลือกไฟล์รูปภาพใบมันสำปะหลัง</li>
                <li>📍 เลือกตำแหน่งที่ถ่ายภาพ </li>
                <li>📅 เลือกวันที่ถ่ายภาพ</li>
                <li>🤖 รอระบบ AI วิเคราะห์</li>
                <li>📊 ดูผลการวิเคราะห์และคำแนะนำ</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚠️ คำเตือน")
        st.markdown("""
        <div class="status-warning">
            ไม่แนะนำให้ใช้กับรูปภาพอื่นนอกจากใบมันสำปะหลัง และถ่ายใบใกล้ๆ
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
        model, device = load_model()

    if model is None:
        st.markdown("""
        <div class="status-error">
            ❌ ไม่สามารถโหลดโมเดลได้ กรุณาลองใหม่อีกครั้ง
        </div>
        """, unsafe_allow_html=True)
        return

    rules_df = load_weather_rules()
    if rules_df.empty:
        st.markdown("""
        <div class="status-warning">
            ⚠️ ไม่สามารถโหลดกฎเกณฑ์สภาพอากาศได้ การวิเคราะห์สภาพอากาศจะไม่ทำงาน
        </div>
        """, unsafe_allow_html=True)

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

    latitude, longitude = location_input_section()
    # Make date input user-friendly with better validation
    today = datetime.now().date()
    min_date = today - timedelta(days=7)
    analysis_date = st.date_input(
        "วันที่ถ่ายภาพ (ย้อนหลังได้ 7 วัน)", 
        value=today,
        max_value=today,
        min_value=min_date,
        help="เลือกวันที่ถ่ายภาพ (สามารถเลือกได้เฉพาะ 7 วันย้อนหลังจากวันนี้)"
    )
    
    # Convert date to string for API call
    analysis_date_str = analysis_date.strftime("%Y-%m-%d")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Fix image orientation before displaying
        image = fix_image_orientation(image)
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
            
            # --- New: Weather Analysis Section ---
            st.markdown("---")
            st.markdown("## ☁️ การวิเคราะห์สภาพอากาศร่วม")
            if latitude and longitude and analysis_date_str:
                with st.spinner("กำลังดึงข้อมูลสภาพอากาศและวิเคราะห์..."):
                    # ดึงข้อมูลสภาพอากาศปัจจุบัน
                    current_weather, current_error = get_current_weather(latitude, longitude)
                    
                    if current_error:
                        st.error(f"⚠️ ไม่สามารถดึงข้อมูลสภาพอากาศปัจจุบัน: {current_error}")
                    elif current_weather:
                        st.write("### 🌤️ สภาพอากาศปัจจุบัน")
                        st.write(f"*อัพเดทล่าสุด: {current_weather['last_updated']}*")
                        
                        # แสดงข้อมูลสภาพอากาศปัจจุบัน
                        curr_col1, curr_col2, curr_col3 = st.columns(3)
                        with curr_col1:
                            st.metric("🌡️ อุณหภูมิ", f"{current_weather['temp_c']:.1f}°C")
                        with curr_col2:
                            st.metric("💧 ความชื้นสัมพัทธ์", f"{current_weather['humidity']}%")
                        with curr_col3:
                            st.metric("🌧️ ปริมาณน้ำฝน", f"{current_weather['precip_mm']:.1f}มม.")
                        
                        # แสดงข้อมูลเพิ่มเติม
                        extra_col1, extra_col2 = st.columns(2)
                        with extra_col1:
                            st.metric("💨 ความเร็วลม", f"{current_weather['wind_kph']} กม./ชม.")
                        with extra_col2:
                            st.write("☁️ สภาพอากาศ:", current_weather['condition'])

                    # ดึงข้อมูลย้อนหลัง
                    historical_weather, error_msg = get_historical_weather(latitude, longitude, analysis_date_str, days_back=7)

                    if error_msg:
                        st.error(f"⚠️ {error_msg}")
                    elif historical_weather:
                        st.write("### 📊 ข้อมูลสภาพอากาศย้อนหลัง 7 วัน")
                        
                        # Calculate average/total for the past 7 days for relevant parameters
                        temps = [d.get('avgtemp_c') for d in historical_weather.values() if d.get('avgtemp_c') is not None]
                        humidities = [d.get('avghumidity') for d in historical_weather.values() if d.get('avghumidity') is not None]
                        precipitations = [d.get('totalprecip_mm') for d in historical_weather.values() if d.get('totalprecip_mm') is not None]

                        if not temps or not humidities or not precipitations:
                            st.warning("⚠️ ข้อมูลสภาพอากาศบางส่วนไม่สมบูรณ์ ผลการวิเคราะห์อาจไม่แม่นยำ")

                        avg_temp_7_days = np.mean(temps) if temps else None
                        avg_humidity_7_days = np.mean(humidities) if humidities else None
                        total_precip_7_days = np.sum(precipitations) if precipitations else None

                        # Display weather data
                        st.write("### ข้อมูลสภาพอากาศ")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🌡️ อุณหภูมิเฉลี่ย", f"{avg_temp_7_days:.1f}°C" if avg_temp_7_days is not None else "ไม่มีข้อมูล")
                        with col2:
                            st.metric("💧 ความชื้นสัมพัทธ์เฉลี่ย", f"{avg_humidity_7_days:.1f}%" if avg_humidity_7_days is not None else "ไม่มีข้อมูล")
                        with col3:
                            st.metric("🌧️ ปริมาณน้ำฝนรวม", f"{total_precip_7_days:.1f}มม." if total_precip_7_days is not None else "ไม่มีข้อมูล")

                        # Prepare summary for rule evaluation
                        weather_data_summary = {
                            'avg_temp_c': avg_temp_7_days,
                            'avg_humidity': avg_humidity_7_days,
                            'total_precip_mm': total_precip_7_days,
                        }
                        
                        if not rules_df.empty and predicted_class is not None:
                            weather_messages = evaluate_weather_rules(predicted_class, weather_data_summary, rules_df)
                            for msg in weather_messages:
                                st.markdown(msg)
                        else:
                            st.info("ไม่พบกฎเกณฑ์สภาพอากาศสำหรับวิเคราะห์เพิ่มเติม")
                    else:
                        st.warning("ไม่สามารถดึงข้อมูลสภาพอากาศย้อนหลังได้ กรุณาตรวจสอบละติจูด/ลองจิจูด และ WeatherAPI Key ของคุณ")
            else:
                st.info("โปรดระบุละติจูด ลองจิจูด และวันที่ถ่ายภาพ เพื่อรับการวิเคราะห์สภาพอากาศเพิ่มเติม")
            # --- End Weather Analysis Section ---


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
                st.markdown("### 📸 รูปภาพตัวอย่าง")
                display_image_slider(info['example_images'], f"ตัวอย่างอาการ {info['thai_name']}")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🌾 ระบบวิเคราะห์โรคมันสำปะหลัง <br>
        ⚠️ <em>หมายเหตุ: ผลการวิเคราะห์เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาผู้เชี่ยวชาญเพิ่มเติม</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()