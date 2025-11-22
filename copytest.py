import shutil
import os
if True:
    filename = os.path.abspath("tests/20231714422*1e79377f-a8b2-4e42-9389-0de410c76e0c_4.wav")
    destination = "/root/duix_avatar_data/face2face/audio/"
    try:
        shutil.copy2(filename, destination)
        print(f"文件 {filename} 已成功复制到 {destination}")
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，无法复制。")
    except PermissionError:
        print(f"没有权限将文件 {filename} 复制到 {destination}。")
    except Exception as e:
        print(f"复制文件时出现错误: {e}")
