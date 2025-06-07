import os
import warnings
warnings.filterwarnings('ignore')
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
import glob
import requests
from datetime import datetime, timedelta
import json

# --- Config and CSS ---
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

# --- Disease Info (unchanged) ---
DISEASE_INFO = {
    'CBB': {
        'name': 'CBB (Cassava Bacterial Blight)',
        'thai_name': 'โรคใบไหม้มันสำปะหลัง',
        'description': 'โรคที่เกิดจากแบคทีเรีย ทำให้ใบมันสำปะหลังเหี่ยวเฉาและตาย',
        'symptoms': ' แสดงอาการใบจุดเหลี่ยมฉํ่านํ้า ใบไหม้ ใบเหี่ยว ยางไหลจนถึงอากยอดเหี่ยวและแห้งตายลงมา นอกจากนี้ยังทำให้ระบบท่อนํ้าอาหารของลำต้นและรากเน่า',
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
        'symptoms': 'ใบเกิดการบิดเบี้ยว และเจริญเติบโตช้า ต้นไม้จะตายได้หากโรครุนแรง',
        'treatment': 'ใช้พันธุ์มันสำปะหลังต้านทานโรคพืช, กำจัดแมลงหวี่ขาว',
        'severity': 'ต่ำ',
        'emoji': '🦠',
        'example_images': sorted(glob.glob('streamlit/assets/img/CGM/*.jpg') + glob.glob('streamlit/assets/img/CGM/*.png'))
    },
    'CMD': {
        'name': 'CMD (Cassava Mosaic Disease)',
        'thai_name': 'โรคใบด่างมันสำปะหลัง',
        'description': 'ทำให้ต้นมันสำปะหลังมีอาการใบด่างเหลือง ใบเสียรูปทรง หดลดรูป',
        'symptoms': 'ลำต้นแคระแกร็น ไม่เจริญเติบโต หรือมีการเจริญเติบโตน้อย ต้นมันสำปะหลังไม่สร้างหัว',
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

# --- Model Definitions (unchanged) ---
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
        
        if mean_intensity < 50 or mean_intensity > 200: # Thresholds for very dark/bright images
            return False, None, None
        
        # Simple entropy check for image richness
        entropy = -np.sum((gray_image / 255.0) * np.log2(gray_image / 255.0 + 1e-10))
        
        if entropy < 3.0: # Threshold for low entropy (e.g., plain background)
            return False, None, None
        
        return True, None, entropy # confidence_score is not used in this validation
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตรวจสอบรูปภาพ: {str(e)}")
        return False, None, None

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        model = vit_base_patch32_model(num_classes=5)
        model_path = "streamlit/assets/model/best_model.pth" # Adjust path if necessary
        
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

# --- NEW: Geolocation function ---
def get_location_js():
    js_code = """
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    window.parent.postMessage({
                        streamlit: {
                            command: "SET_VALUE",
                            args: {
                                key: "geolocation_data",
                                value: JSON.stringify({lat: lat, lon: lon})
                            }
                        }
                    }, "*");
                },
                (error) => {
                    window.parent.postMessage({
                        streamlit: {
                            command: "SET_VALUE",
                            args: {
                                key: "geolocation_data",
                                value: JSON.stringify({error: error.message})
                            }
                        }
                    }, "*");
                },
                {
                    enableHighAccuracy: true,
                    timeout: 5000,
                    maximumAge: 0
                }
            );
        } else {
            window.parent.postMessage({
                streamlit: {
                    command: "SET_VALUE",
                    args: {
                        key: "geolocation_data",
                        value: JSON.stringify({error: "Geolocation is not supported by this browser."})
                    }
                }
            }, "*");
        }
    }
    getLocation();
    """
    # This workaround is needed to execute JS in Streamlit custom component contexts
    # It might require 'streamlit_javascript' or similar if direct js injection is problematic
    # For now, we'll try with st.components.v1.html directly or assume a custom component for simplicity
    st.components.v1.html(f"<script>{js_code}</script>", height=0, width=0)

# --- NEW: Weather API Function ---
WEATHER_API_KEY = "2e8cfa89ce124ecca55102846250706" # <<<--- ** ใส่ API Key ของคุณที่นี่ **
WEATHER_API_URL = "http://api.weatherapi.com/v1/history.json"

@st.cache_data(ttl=timedelta(hours=1)) # Cache weather data for 1 hour to avoid repeated API calls
def get_historical_weather(latitude, longitude, date_str, days_back=7):
    historical_data = {}
    
    for i in range(days_back):
        target_date = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=i)
        date_param = target_date.strftime("%Y-%m-%d")
        
        params = {
            "key": WEATHER_API_KEY,
            "q": f"{latitude},{longitude}",
            "dt": date_param
        }
        
        try:
            response = requests.get(WEATHER_API_URL, params=params)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            
            if "forecast" in data and "forecastday" in data["forecast"] and len(data["forecast"]["forecastday"]) > 0:
                day_data = data["forecast"]["forecastday"][0]["day"]
                hour_data = data["forecast"]["forecastday"][0]["hour"]

                historical_data[date_param] = {
                    "avgtemp_c": day_data.get("avgtemp_c"),
                    "maxwind_kph": day_data.get("maxwind_kph"),
                    "totalprecip_mm": day_data.get("totalprecip_mm"),
                    "avghumidity": day_data.get("avghumidity"),
                    "daily_chance_of_rain": day_data.get("daily_chance_of_rain"),
                    # You might need to calculate avg for other parameters if not directly available
                    # For example, average of hourly temps
                    "hourly_temp_c": [h.get("temp_c") for h in hour_data if h.get("temp_c") is not None],
                    "hourly_humidity": [h.get("humidity") for h in hour_data if h.get("humidity") is not None]
                }
            else:
                st.warning(f"ไม่พบข้อมูลสภาพอากาศสำหรับวันที่ {date_param} ที่พิกัด {latitude},{longitude}")
                historical_data[date_param] = {} # Store empty dict for this date if no data
        except requests.exceptions.RequestException as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลสภาพอากาศสำหรับวันที่ {date_param}: {e}")
            historical_data[date_param] = {} # Store empty dict for this date if error
            
    return historical_data

# --- NEW: Load and Process Rules ---
@st.cache_data
def load_weather_rules(file_path="/streamlit/assets/weather_rules.csv"):
    try:
        rules_df = pd.read_csv(file_path)
        # Clean column names for easier access
        rules_df.columns = [col.strip().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '') for col in rules_df.columns]
        return rules_df
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์กฎเกณฑ์สภาพอากาศ: {file_path} กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับ app.py")
        return pd.DataFrame() # Return empty DataFrame if file not found

# --- NEW: Evaluate Weather Rules ---
def evaluate_weather_rules(predicted_disease_key, weather_data_summary, rules_df):
    relevant_rules = rules_df[rules_df['โรค_สภาวะ'] == predicted_disease_key]
    
    if relevant_rules.empty:
        return [] # No specific rules for this disease

    messages = []
    
    for index, row in relevant_rules.iterrows():
        param = row['พารามิเตอร์สภาพอากาศหลัก'].strip().lower()
        condition_str = str(row['ช่วงที่เหมาะสม_เอื้ออำนวย_สำหรับโรค_สุขภาพ_']).strip()
        note = str(row['หมายเหตุ_เงื่อนไขเฉพาะ']).strip()
        risk_level = str(row['ระดับความเสี่ยง_สถานะ']).strip()

        # Parse condition string (simple parser for common operators)
        operator = None
        value = None
        if '>' in condition_str:
            operator = '>'
            value = float(condition_str.replace('>', '').replace('%', '').replace('C', '').replace('มม._วัน', '').strip())
        elif '>=' in condition_str:
            operator = '>='
            value = float(condition_str.replace('>=', '').replace('%', '').replace('C', '').replace('มม._วัน', '').strip())
        elif '<' in condition_str:
            operator = '<'
            value = float(condition_str.replace('<', '').replace('%', '').replace('C', '').replace('มม._วัน', '').strip())
        elif '<=' in condition_str:
            operator = '<='
            value = float(condition_str.replace('<=', '').replace('%', '').replace('C', '').replace('มม._วัน', '').strip())
        elif '-' in condition_str: # Range, e.g., "22-26°C"
            parts = condition_str.split('-')
            if len(parts) == 2:
                try:
                    min_val = float(parts[0].strip().replace('C', '').replace('มม.', ''))
                    max_val = float(parts[1].strip().replace('C', '').replace('มม.', ''))
                    operator = 'range'
                    value = (min_val, max_val)
                except ValueError:
                    pass # Fallback if parsing fails
        else: # Direct value comparison or specific keywords
            try:
                value = float(condition_str.replace('%', '').replace('C', '').replace('มม.', '').strip())
                operator = '=' # Assume equality if no operator
            except ValueError:
                # Handle cases like "สูง" or "ต่ำ" or "การเริ่มต้นฤดูฝน" directly
                if condition_str == 'สูง': operator = 'high_kw'
                elif condition_str == 'ต่ำ': operator = 'low_kw'
                elif condition_str == 'ช่วงแห้งแล้งยาวนาน': operator = 'dry_long'
                elif condition_str == 'การเริ่มต้นฤดูฝน': operator = 'rainy_start'
                elif condition_str == 'เหมาะสม': operator = 'optimal'
                elif condition_str == 'ความชื้นสูง': operator = 'high_humidity_kw'
                elif condition_str == 'แสงแดดโดยตรง': operator = 'direct_sun_kw'
                elif condition_str == 'ระบายน้ำดี, อุดมสมบูรณ์, pH 5.5-6.5': operator = 'soil_optimal_kw'

        is_match = False
        current_param_value = None

        # Map spreadsheet param names to weather_data_summary keys
        weather_param_map = {
            'ความชื้นสัมพัทธ์': 'avg_humidity', # Renamed in summary
            'อุณหภูมิ': 'avg_temp_c',
            'ปริมาณน้ำฝน': 'total_precip_mm',
            'ความชื้น': 'avg_humidity', # Duplicate for flexibility
            'แสงแดดโดยตรง': 'avg_sunshine_hours', # Need to get this from API if available
            'สภาพดิน': 'soil_data_placeholder' # This cannot be derived from weather API
        }

        if param in weather_param_map and weather_param_map[param] in weather_data_summary:
            current_param_value = weather_data_summary[weather_param_map[param]]

            if operator == '>': is_match = current_param_value > value
            elif operator == '>=': is_match = current_param_value >= value
            elif operator == '<': is_match = current_param_value < value
            elif operator == '<=': is_match = current_param_value <= value
            elif operator == '=' and isinstance(value, float): is_match = current_param_value == value
            elif operator == 'range' and isinstance(value, tuple):
                is_match = value[0] <= current_param_value <= value[1]
            # Handle keyword operators for broader categories
            elif operator == 'high_kw' and param == 'ปริมาณน้ำฝน':
                is_match = current_param_value > 5 # Example threshold for "high" rainfall
            elif operator == 'dry_long' and param == 'สภาพอากาศ':
                is_match = current_param_value < 10 # Example threshold for "low" rainfall in a period
            elif operator == 'optimal' and param == 'อุณหภูมิ' and 25 <= current_param_value <= 32:
                 is_match = True
            elif operator == 'optimal' and param == 'ความชื้นสัมพัทธ์' and 70 <= current_param_value <= 80:
                 is_match = True
            # Add more specific keyword mappings as needed for other params

        # Special handling for "สภาพอากาศ" (general climate conditions)
        if param == 'สภาพอากาศ' and 'ปริมาณน้ำฝน' in weather_data_summary:
            if operator == 'ช่วงแห้งแล้งยาวนาน' and weather_data_summary['total_precip_mm'] < 10: # Example low rainfall for dry
                is_match = True
            elif operator == 'การเริ่มต้นฤดูฝน' and weather_data_summary['total_precip_mm'] > 20: # Example high rainfall for rainy season start
                is_match = True

        # Special handling for notes/conditions from the sheet (e.g., "ต้องมีระยะเวลา ≥12 ชั่วโมง")
        # This part is complex to automate purely with current weather API data.
        # For simplicity, we'll just check if the main condition is met.
        
        if is_match:
            messages.append(f"**{row['ระดับความเสี่ยง_สถานะ']}** จากพารามิเตอร์ **{row['พารามิเตอร์สภาพอากาศหลัก']}** ({current_param_value:.1f} {condition_str.split(' ')[-1]}): {note}")
    
    return messages

# --- Main Streamlit App ---
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
    3. 📍 ระบบจะพยายามดึงตำแหน่งอัตโนมัติ
    4. 🌦️ ดึงข้อมูลสภาพอากาศในพื้นที่นั้นๆ
    5. 📊 ดูผลการวิเคราะห์และคำแนะนำเพิ่มเติมจากสภาพอากาศ
    """)

    with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
        model, device = load_model()

    if model is None:
        st.error("❌ ไม่สามารถโหลดโมเดลได้")
        return

    # --- Geolocation Button and Logic ---
    st.markdown("## 📍 ข้อมูลตำแหน่งของคุณ")
    location_placeholder = st.empty()

    if 'location_data' not in st.session_state:
        st.session_state.location_data = None
        st.button("คลิกเพื่อดึงตำแหน่งอัตโนมัติ (จำเป็นต้องอนุญาต)", on_click=get_location_js)
    
    if st.session_state.location_data:
        loc_data = st.session_state.location_data
        if isinstance(loc_data, str): # if it's a JSON string from JS
            try:
                loc_data = json.loads(loc_data)
                st.session_state.location_data = loc_data # update session state with parsed dict
            except json.JSONDecodeError:
                st.error("เกิดข้อผิดพลาดในการประมวลผลข้อมูลตำแหน่ง")
                st.session_state.location_data = None # Reset to try again

        if st.session_state.location_data and "lat" in st.session_state.location_data:
            st.success(f"✔️ พบตำแหน่งของคุณ: ละติจูด {st.session_state.location_data['lat']:.4f}, ลองจิจูด {st.session_state.location_data['lon']:.4f}")
            st.caption("ข้อมูลตำแหน่งถูกดึงอัตโนมัติจากเบราว์เซอร์ของคุณ")
        elif st.session_state.location_data and "error" in st.session_state.location_data:
            st.error(f"❌ ไม่สามารถดึงตำแหน่งได้: {st.session_state.location_data['error']}. โปรดอนุญาตการเข้าถึงตำแหน่งในเบราว์เซอร์ของคุณ หรือลองรีเฟรชหน้า.")
            st.session_state.location_data = None # Allow retry
        else:
             st.info("รอการดึงตำแหน่งอัตโนมัติ... โปรดกดปุ่มด้านบนและอนุญาต")
    else:
        st.info("โปรดกดปุ่ม 'คลิกเพื่อดึงตำแหน่งอัตโนมัติ' และอนุญาตในเบราว์เซอร์ของคุณ เพื่อให้ระบบดึงข้อมูลสภาพอากาศได้")

    st.markdown("---")


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
            # For camera_input, we assume current date for weather data
            st.session_state.photo_date = datetime.now().strftime("%Y-%m-%d")
        elif uploaded_file is None:
            # If no camera image and no file uploaded yet, set default date for potential past uploads
            st.session_state.photo_date = datetime.now().strftime("%Y-%m-%d")
        
    
    # Allow user to manually input photo date if not from camera
    if uploaded_file is not None and upload_option == "📁 เลือกไฟล์จากเครื่อง":
        st.session_state.photo_date = st.date_input(
            "วันที่ถ่ายภาพ (โดยประมาณ)",
            value=datetime.now(),
            max_value=datetime.now() # Cannot be a future date
        ).strftime("%Y-%m-%d")
    elif uploaded_file is None:
        # Placeholder for date if no file yet
        st.session_state.photo_date = datetime.now().strftime("%Y-%m-%d")


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
                if not is_valid:
                    st.error("❌ ไม่สามารถระบุได้ว่าเป็นรูปใบมันสำปะหลัง หรือรูปภาพไม่ชัดเจนพอ")
                    st.warning("""
                        ⚠️ กรุณาตรวจสอบว่า:
                        - รูปภาพเป็นใบมันสำปะหลังจริง
                        - รูปภาพชัดเจน ไม่เบลอ, ไม่มืด/สว่างเกินไป
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
                
                # --- NEW: Weather Data Display and Rules Evaluation ---
                st.markdown("---")
                st.markdown("### ☁️ ข้อมูลสภาพอากาศในพื้นที่")
                if st.session_state.location_data and "lat" in st.session_state.location_data:
                    latitude = st.session_state.location_data['lat']
                    longitude = st.session_state.location_data['lon']
                    
                    with st.spinner(f"กำลังดึงข้อมูลสภาพอากาศย้อนหลัง 7 วัน สำหรับวันที่ {st.session_state.photo_date}..."):
                        historical_weather = get_historical_weather(latitude, longitude, st.session_state.photo_date, days_back=7)
                    
                    if historical_weather:
                        # Calculate average for the past 7 days for relevant parameters
                        temps = [d.get('avgtemp_c') for d in historical_weather.values() if d.get('avgtemp_c') is not None]
                        humidities = [d.get('avghumidity') for d in historical_weather.values() if d.get('avghumidity') is not None]
                        precipitations = [d.get('totalprecip_mm') for d in historical_weather.values() if d.get('totalprecip_mm') is not None]

                        avg_temp_7_days = np.mean(temps) if temps else None
                        avg_humidity_7_days = np.mean(humidities) if humidities else None
                        total_precip_7_days = np.sum(precipitations) if precipitations else None
                        
                        st.write(f"**สภาพอากาศเฉลี่ย 7 วันย้อนหลัง (นับจากวันที่ {st.session_state.photo_date})**")
                        if avg_temp_7_days is not None:
                            st.metric(label="อุณหภูมิเฉลี่ย", value=f"{avg_temp_7_days:.1f} °C")
                        if avg_humidity_7_days is not None:
                            st.metric(label="ความชื้นสัมพัทธ์เฉลี่ย", value=f"{avg_humidity_7_days:.1f} %")
                        if total_precip_7_days is not None:
                            st.metric(label="ปริมาณน้ำฝนรวม", value=f"{total_precip_7_days:.1f} มม.")
                        
                        # Prepare summary for rule evaluation
                        weather_data_summary = {
                            'avg_temp_c': avg_temp_7_days,
                            'avg_humidity': avg_humidity_7_days,
                            'total_precip_mm': total_precip_7_days,
                            # Add more aggregated data as needed by your rules, e.g., max wind, min temp
                        }

                        # Load rules and evaluate
                        weather_rules_df = load_weather_rules()
                        if not weather_rules_df.empty:
                            weather_messages = evaluate_weather_rules(predicted_class, weather_data_summary, weather_rules_df)
                            if weather_messages:
                                st.markdown("#### ✨ คำแนะนำเพิ่มเติมจากสภาพอากาศ:")
                                for msg in weather_messages:
                                    st.markdown(f"- {msg}")
                            else:
                                st.info("ไม่พบเงื่อนไขสภาพอากาศที่ตรงกับโรคนี้ในช่วงเวลาที่ผ่านมา")
                        else:
                            st.warning("ไม่สามารถโหลดกฎเกณฑ์สภาพอากาศได้ กรุณาตรวจสอบไฟล์ weather_rules.csv")
                    else:
                        st.warning("ไม่สามารถดึงข้อมูลสภาพอากาศย้อนหลังได้ โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตหรือ API Key")
                else:
                    st.info("โปรดดึงตำแหน่งของคุณเพื่อดูข้อมูลสภาพอากาศในพื้นที่")

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