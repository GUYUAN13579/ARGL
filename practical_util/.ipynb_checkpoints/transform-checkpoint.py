import os  
import re  

# 定义要处理的文件夹路径  
folder_path = '/workspace/SSDA-YOLO/ship_source/ship_source/shipA_source/shipA_fake/labels/train'  

# 遍历文件夹中的所有文件  
for filename in os.listdir(folder_path):  
    if filename.endswith('.txt'):  
        file_path = os.path.join(folder_path, filename)  
        
        # 读取文件内容  
        with open(file_path, 'r', encoding='utf-8') as file:  
            lines = file.readlines()  
        
        # 替换每一行的第一个数字为0  
        new_lines = []  
        for line in lines:  
            # 使用正则表达式匹配第一个数字  
            new_line = re.sub(r'^\s*\d+', '0', line)  
            new_lines.append(new_line)  

        # 将修改后的内容写回文件  
        with open(file_path, 'w', encoding='utf-8') as file:  
            file.writelines(new_lines)  

print("处理完成!")