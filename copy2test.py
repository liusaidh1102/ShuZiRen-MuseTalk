import os

# 文件名用单引号包裹，避免 * 被解析
filename = "tests/20231714422*1e79377f-a8b2-4e42-9389-0de410c76e0c_4.wav"
destination = "/root/duix_avatar_data/face2face/audio/"

# 构造终端复制命令（用 cp 命令，单引号包裹文件名）
cmd = f"cp '{filename}' '{destination}'"

# 执行命令
ret = os.system(cmd)

if ret == 0:
    print("复制成功")
else:
    print("复制失败，可能文件不存在或权限不足")
