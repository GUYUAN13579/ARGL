import zipfile  
import os  

# 指定压缩文件和解压目标文件夹  
zip_file_path = '/workspace/SSDA-YOLO/supplement2.zip'  
extract_to = '/workspace/SSDA-YOLO/supplement2'

# 创建目标文件夹（如果不存在的话）  
os.makedirs(extract_to, exist_ok=True)  

# 解压 ZIP 文件  
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:  
    zip_ref.extractall(extract_to)  

print(f'解压完成，文件已解压至: {extract_to}')
