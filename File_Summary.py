import os
from collections import defaultdict

# 你的猫咪文件夹路径
cat_folders = [
    r"D:\CAT_IN_HQU\Lid#锅盖",
    r"D:\CAT_IN_HQU\McDonald#麦麦",
    r"D:\CAT_IN_HQU\oreange#瘦橘",
    r"D:\CAT_IN_HQU\Slit#有条缝",
    r"D:\CAT_IN_HQU\tangyuan#汤圆有鱼年年",
    r"D:\CAT_IN_HQU\WhiteRabbit#千岛酱白露兔子",
    r"D:\CAT_IN_HQU\yuan#冤种"
]

# 定义图片和视频扩展名
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
video_exts = {'.mp4', '.avi', '.mov', '.mkv'}


def analyze_folder(folder_path):
    stats = {
        'total_files': 0,
        'total_size_MB': 0,
        'image_count': 0,
        'video_count': 0,
        'file_types': defaultdict(int)
    }

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            stats['total_files'] += 1
            file_path = os.path.join(root, file)
            stats['total_size_MB'] += os.path.getsize(file_path) / (1024 * 1024)

            ext = os.path.splitext(file)[1].lower()
            stats['file_types'][ext] += 1

            if ext in image_exts:
                stats['image_count'] += 1
            elif ext in video_exts:
                stats['video_count'] += 1

    return stats


# 遍历每个猫咪文件夹
for folder in cat_folders:
    cat_name = os.path.basename(folder)
    stats = analyze_folder(folder)
    print(f"猫咪: {cat_name}")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  总大小: {stats['total_size_MB']:.2f} MB")
    print(f"  图片数量: {stats['image_count']}")
    print(f"  视频数量: {stats['video_count']}")
    print(f"  文件类型分布: {dict(stats['file_types'])}")
    print("-" * 50)