import zipfile
import os
import pandas as pd

# 指定zip文件所在的目录
zip_folder = "./AACHEN"

# 创建一个用于保存解压缩文件的目录
unzip_folder = "aaaachen"
if not os.path.exists(unzip_folder):
    os.makedirs(unzip_folder)


# 遍历zip文件并逐个处理
for file_name in os.listdir(zip_folder):
    if file_name.endswith(".zip"):
        # 获取zip文件的路径
        zip_file_path = os.path.join(zip_folder, file_name)

        # 打开zip文件并解压缩
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_folder)
        except:
            pass

