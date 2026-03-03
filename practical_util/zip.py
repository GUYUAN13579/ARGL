# 压缩当前路径所有文件，输出zip文件
path='/workspace/SSDA-YOLO/runs/train/model_with_copypaste3'

import zipfile,os
zipName = '/workspace/SSDA-YOLO/model_with_copypaste4.zip' #压缩后文件的位置及名称
f = zipfile.ZipFile( zipName, 'w', zipfile.ZIP_DEFLATED )
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        print(filename)
        f.write(os.path.join(dirpath,filename))
f.close()

