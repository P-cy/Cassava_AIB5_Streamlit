import streamlit as st
import torch

def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = vit_base_patch32_model(num_classes=5)
        model_path = "assets/model/best_model.pth"  # Updated path
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        checkpoint = remove_module_prefix(checkpoint)

        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบเส้นทางไฟล์")
        return None, None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None, None 