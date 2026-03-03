import shutil
import os


def copy_files(src_folder, dest_folder):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(dest_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 构造完整的源文件路径
        src_file = os.path.join(src_folder, filename)

        # 检查是否为文件而不是文件夹
        if os.path.isfile(src_file):
            # 构造目标文件路径
            dest_file = os.path.join(dest_folder, filename)
            # 复制文件
            shutil.copy2(src_file, dest_file)
            print(f'已复制: {src_file} 到 {dest_file}')

        # 使用示例


src_folder = '/workspace/SSDA-YOLO/supplement2/supplement2/labels'
dest_folder = '/workspace/SSDA-YOLO/ship_source/ship_source/shipA_source/shipA_fake/labels/train'
copy_files(src_folder, dest_folder)  