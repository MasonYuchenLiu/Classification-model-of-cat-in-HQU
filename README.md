# HQU 猫识别器 / HQU Cat Classifier

这是一个基于 **PyTorch + Gradio** 的猫识别项目，可以识别 7 种猫 + 未知。

This is a **PyTorch + Gradio** based cat classification project. It can recognize 7 specific cats plus an "Unknown" category.

---

## 功能 / Features

- 识别 7 种猫 + 未知
- 自动下载模型，无需手动配置
- 本地运行或通过 Gradio 网页界面使用
- 输出类别概率（可选扩展）

- Recognize 7 cats + "Unknown"
- Automatically download the model from GitHub
- Run locally with Gradio web interface
- Optionally display probability scores

---

## 支持的猫类别 / Supported Cats

1. 锅盖 / Lid
2. 麦麦 / McDonald
3. 瘦橘 / Orange
4. 有条缝 / Slit
5. 汤圆有鱼年年 / Tangyuan
6. 千岛酱白露兔子 / WhiteRabbit
7. 冤种 / Yuan
8. 未知 / Unknown

---

## 使用方法 / How to Use

### 1. 克隆仓库 / Clone the repository

```bash
git clone https://github.com/MasonYuchenLiu/Classification-model-of-cat-in-HQU.git
cd Classification-model-of-cat-in-HQU
2. 安装依赖 / Install dependencies

建议使用虚拟环境 / It's recommended to use a virtual environment:

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

安装依赖 / Install required packages:

pip install -r requirements.txt
3. 运行应用 / Run the app
python app.py

运行后 Gradio 会生成一个本地网页链接，通常类似：

Running on local URL:  http://127.0.0.1:7860

在浏览器中打开链接，即可上传猫照片进行识别。

After running, Gradio will provide a local URL (e.g., http://127.0.0.1:7860). Open it in your browser and upload cat images for recognition.

模型 / Model
模型会在第一次运行时自动下载到本地，无需手动下载
Model will be automatically downloaded on first run

下载地址 / Download URL:

https://github.com/MasonYuchenLiu/Classification-model-of-cat-in-HQU/raw/main/cat_classifier.pth
注意事项 / Notes
确保网络可访问 GitHub，以便自动下载模型
如果模型文件过大，可考虑缓存或本地保存
运行 app.py 时，请确保 Python 版本 >=3.8
Make sure you can access GitHub to download the model automatically
If the model is large, consider caching locally
Python version >=3.8 is recommended
联系 / Contact

如有问题，请在 GitHub Issues 提问。
If you have questions, please open an issue on GitHub.
