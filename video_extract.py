import os
import cv2

# 猫咪文件夹路径
cat_folders = [
    r"D:\CAT_IN_HQU\Lid#锅盖",
    r"D:\CAT_IN_HQU\McDonald#麦麦",
    r"D:\CAT_IN_HQU\oreange#瘦橘",
    r"D:\CAT_IN_HQU\Slit#有条缝",
    r"D:\CAT_IN_HQU\tangyuan#汤圆有鱼年年",
    r"D:\CAT_IN_HQU\WhiteRabbit#千岛酱白露兔子",
    r"D:\CAT_IN_HQU\yuan#冤种"
]

video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
frame_rate = 240


def extract_frames(video_path, save_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps * frame_rate)  # 每隔多少帧抽一帧

    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            filename = os.path.join(save_folder,
                                    f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        count += 1
    cap.release()
    print(f"{video_path} 提取 {saved_count} 帧完成")


for folder in cat_folders:
    frames_folder = os.path.join(folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)

    for file in os.listdir(folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in video_exts:
            video_path = os.path.join(folder, file)
            extract_frames(video_path, frames_folder, frame_rate)