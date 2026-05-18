import torch
from torchvision import transforms
from PIL import Image
import requests
import gradio as gr
from model import CatCNN

# 猫名字典
CAT_LABELS = {
    0: "锅盖",
    1: "麦麦",
    2: "瘦橘",
    3: "有条缝",
    4: "汤圆有鱼年年",
    5: "千岛酱白露兔子",
    6: "冤种"
}

# 模型下载与加载
MODEL_URL = "https://github.com/MasonYuchenLiu/Classification-model-of-cat-in-HQU/raw/refs/heads/main/cat_classifier.pth"
MODEL_PATH = "cat_classifier.pth"

def download_model():
    import os
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

download_model()
model = CatCNN(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
model.eval()

# 猫识别函数
def predict_cat(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probs, 1)
        pred_class = pred_class.item()
        if max_prob < 0.6:
            return "未知"
        return CAT_LABELS.get(pred_class, "未知")

# Gradio 界面
iface = gr.Interface(
    fn=predict_cat,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="识别结果"),
    title="HQU 猫识别器",
    description="上传一张猫的照片，识别猫的类别（7种+未知）"
)

iface.launch()