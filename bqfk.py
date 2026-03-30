import os
import redis
import mysql.connector
import cv2
import requests
import json

# # Dify API 相关信息
# 表情动作反馈
DIFY_API_KEY = "app-zYVvuG0iSbAeubED4xnfHWiw"
DIFY_API_URL = "https://api.dify.ai/v1/chat-messages"
# 本地文件nginx代理
BASE_URL = "http://localhost:4000/tests/"
# 容器内部访问外部的地址
REMOTE_BASE_URL = "http://localhost:4000/tests/"
TESTS_DIR = r"tests"

r = redis.Redis(host='localhost', port=6379, password='123456')
# 视频抽帧函数，抽成图片，并且交给dify进行ai分析
def extract_frames(element, interval=100):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="shuziren-mianshi"
    )
    cursor = connection.cursor()
    # 视频地址，msId，index
    data = element.split("___")
    # OpenCV 读取视频 / 摄像头, 视频地址
    cap = cv2.VideoCapture(BASE_URL + data[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频fps: {fps}")
    if not fps or fps <= 0:
        print(f"视频无法打开或FPS为0: {BASE_URL + data[0]}")
        return
    # fps：1s播放多少图片
    # interval = 你想 多少秒抽一张图（比如 1 秒、2 秒）
    # 目前是 100s 抽一张图片
    frame_interval = max(1, int(fps * interval))

    print(f"每隔{interval} 秒抽一张图片")
    frame_count = 0
    folder_path = os.path.join(TESTS_DIR, data[0].replace('.mp4', ''))
    os.makedirs(folder_path, exist_ok=True)
    print(f"抽帧保存目录: {folder_path}")

    while cap.isOpened():
        # ret成功读取到了，frame为当前帧
        ret, frame = cap.read()
        if not ret:
            break
        # 采用间隔抽帧，第0帧一定会被抽出来
        if frame_count % frame_interval == 0:
            filename = f"{data[0].replace('.mp4', '')}/{frame_count}.jpg"
            frame_filename = os.path.join(TESTS_DIR, filename)
            print(f"抽帧图片地址：{frame_filename}")
            # 图片写入本地磁盘
            cv2.imwrite(frame_filename, frame)
            # 图片地址
            res = analyze_image(REMOTE_BASE_URL + filename)
            print(f"分析结果：{res}") #res['score'] res['summary']
            if "人物" not in str(res['summary']):
                sql = "INSERT INTO b_mianshi_bq (ms_id, `index`, content, sort, url) VALUES (%s, %s, %s, %s, %s)"
                values = (data[1], data[2], str(res['summary']), str(res['score']), filename)
                cursor.execute(sql, values)
                connection.commit()
                print(cursor.rowcount, "条记录插入成功。")
                break
            # extracted_frames.append(frame_filename)
        frame_count += 1

    cap.release()

    cursor.close()
    connection.close()
    # return extracted_frames



# 调用 Dify API 解析图片
def analyze_image(image_path):
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    # with open(image_path, "rb") as f:
        # 这里需要根据 Dify API 文档调整请求体
    payload = {
        "inputs": {},
        "query": "请分析人物的表情动作",
        "response_mode": "blocking",
        "user": "python-biaoqingfankui",
        "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": image_path
            }
        ]
    }
    response = requests.post(DIFY_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return json.loads(result['answer'])

    else:
        print(f"API 请求失败，状态码: {response.status_code}，错误信息: {response.text}")
        return None

while True:
    queue_result = r.blpop('bqfkqueue1', timeout=0)
    element = ""
    if queue_result:
         # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
        element = str(queue_result[1].decode('utf-8'))
        print(f"从队列中取出元素: {element}")
    else:
        print("队列为空，继续等待...")
        continue
    try:
        extract_frames(element)
    except Exception as e:
        print("错误", e)
        import traceback
        traceback.print_exc()
    #print(result)
    #r.set(element, result)

