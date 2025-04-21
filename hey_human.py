import redis
import requests
import shutil
import time

r = redis.Redis(host='10.23.32.63', port=6389, password=None)
r.delete('hey_human')
while True:
    queue_result = r.blpop('hey_human', timeout=0)
    element = ""
    if queue_result:
         # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
        element = str(queue_result[1].decode('utf-8')).split("___")
        key = element[0]
        index = element[1]
        print(f"从队列中取出元素: {element}")
    else:
        print("队列为空，继续等待...")
        continue
    filename = "tests/" + key + "_" + index  + ".wav"
    destination = "/root/heygem_data/face2face/audio/"
    try:
        shutil.copy2(filename, destination)
        print(f"文件 {filename} 已成功复制到 {destination}")
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，无法复制。")
    except PermissionError:
        print(f"没有权限将文件 {filename} 复制到 {destination}。")
    except Exception as e:
        print(f"复制文件时出现错误: {e}")
    url = "http://127.0.0.1:8383/easy/submit"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "audio_url": "/code/data/audio/" + key + "_" + index  + ".wav",
        "video_url": "/code/data/muted_video.mp4",
        "code": key + "_" + index,
        "chaofen": 0,
        "watermark_switch": 0,
        "pn": 1
    }
    print(data)

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")

    while True:
        time.sleep(5)
        try:
            result = requests.get("http://127.0.0.1:8383/easy/query", params={"code": key + "_" + index}).json()
            print(result)
            print(result.get("data", {}))
            msg = result.get("data", {}).get("msg")
            print("任务已完成。" if msg == "任务完成" else f"任务未完成，当前消息: {msg}")
            if msg == "任务完成":
                shutil.copy2("/root/heygem_data/face2face/temp/"+key + "_" + index + "-r.mp4", "/home/sensoro/project/human_ms/tests/" +key + "_" + index + ".mp4")
                r.psetex(key + index + "ok", 360000, 1)
                break
        except (requests.RequestException, ValueError) as err:
            print(f"请求或解析出错: {err}")
