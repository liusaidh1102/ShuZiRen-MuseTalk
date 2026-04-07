import os
import redis
import mysql.connector
import cv2
import requests
import json
from config import DIFY_API_KEY_EXPRESSION_FEEDBACK, DIFY_API_URL

# Redis 配置与规范
r = redis.Redis(host='localhost', port=6379, password='123456', decode_responses=True)
QUEUE_BQFK = 'interview:queue:bqfk'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
CHANNEL_NOTIFY = "interview:channel:notify"

# Dify API 相关信息
DIFY_API_KEY = DIFY_API_KEY_EXPRESSION_FEEDBACK
# 本地文件nginx代理
BASE_URL = "http://localhost:4000/tests/"
# 容器内部访问外部的地址
REMOTE_BASE_URL = "http://localhost:4000/tests/"
TESTS_DIR = r"tests"

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "shuziren-mianshi"
}

def analyze_image(image_url):
    """调用 Dify API 解析图片"""
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": {},
        "query": "请分析人物的表情动作",
        "response_mode": "blocking",
        "user": "python-biaoqingfankui",
        "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": image_url
            }
        ]
    }
    try:
        response = requests.post(DIFY_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return json.loads(result['answer'])
    except Exception as e:
        print(f"[BQFK ERROR] Dify API 请求失败: {e}")
    return None

def extract_and_analyze(task_id, user_id, element, interval=100):
    """视频抽帧并分析"""
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    
    # element 格式: videoUrl___msId___index
    data = element.split("___")
    video_filename = data[0]
    ms_id = data[1]
    index = data[2]

    cap = cv2.VideoCapture(BASE_URL + video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"[BQFK ERROR] 视频无法打开: {BASE_URL + video_filename}")
        return

    frame_interval = max(1, int(fps * interval))
    frame_count = 0
    folder_path = os.path.join(TESTS_DIR, video_filename.replace('.mp4', ''))
    os.makedirs(folder_path, exist_ok=True)

    final_result = {"status": "NO_HUMAN_DETECTED"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rel_filename = f"{video_filename.replace('.mp4', '')}/{frame_count}.jpg"
            frame_filename = os.path.join(TESTS_DIR, rel_filename)
            cv2.imwrite(frame_filename, frame)
            
            res = analyze_image(REMOTE_BASE_URL + rel_filename)
            print(f"[BQFK] 抽帧分析结果: {res}")
            
            if res and "人物" not in str(res.get('summary', '')):
                sql = "INSERT INTO b_mianshi_bq (ms_id, `index`, content, sort, url) VALUES (%s, %s, %s, %s, %s)"
                values = (ms_id, index, str(res['summary']), str(res['score']), rel_filename)
                cursor.execute(sql, values)
                connection.commit()
                final_result = res
                break
        frame_count += 1

    cap.release()
    cursor.close()
    connection.close()

    # 1. 更新任务状态为 DONE
    r.setex(TASK_STATUS.format(taskId=task_id), 600, "DONE")

    # 2. 存储结果到 Redis
    r.setex(TASK_RESULT.format(taskId=task_id), 600, json.dumps(final_result))

    # 3. 发布通知 (Java 端会根据类型过滤，静默入库)
    notify_data = {
        "userId": user_id,
        "taskId": task_id,
        "type": "BQFK",
        "result": final_result
    }
    r.publish(CHANNEL_NOTIFY, json.dumps(notify_data))
    print(f"[BQFK] 任务 {task_id} 处理完毕")

print(f"[BQFK Worker] 启动，正在监听队列: {QUEUE_BQFK}")

while True:
    try:
        queue_result = r.blpop(QUEUE_BQFK, timeout=0)
        if not queue_result:
            continue
            
        task_json = queue_result[1]
        task_data = json.loads(task_json)
        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        element = task_data.get("data") # videoUrl___msId___index

        print(f"[BQFK] 收到任务 {task_id}, 用户 {user_id}")
        extract_and_analyze(task_id, user_id, element)

    except Exception as e:
        print(f"[BQFK ERROR] 循环异常: {e}")
        import traceback
        traceback.print_exc()
        import time
        time.sleep(1)
