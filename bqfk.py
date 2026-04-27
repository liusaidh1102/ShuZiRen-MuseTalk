import os
import pika
import mysql.connector
import cv2
import requests
import json
from config import DIFY_API_KEY_EXPRESSION_FEEDBACK, DIFY_API_URL

# RabbitMQ 配置与规范
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'admin'
RABBITMQ_PASS = '123456'
RABBITMQ_VHOST = 'ai-mianshi'
QUEUE_BQFK = 'interview.queue.bqfk'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
NOTIFY_EXCHANGE = "interview.notify.exchange"
NOTIFY_ROUTING_KEY = "notify.result"

# Dify API 相关信息
DIFY_API_KEY = DIFY_API_KEY_EXPRESSION_FEEDBACK
# 代理mp4视频
BASE_URL = "http://localhost:5000/tests/bqfk/"
# 代理抽帧后的图片
REMOTE_BASE_URL = "http://d49aebc.r35.cpolar.top/tests/frame/"
TESTS_DIR = r"tests"
FRAME_DIR = os.path.join(TESTS_DIR, "frame")

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "shuziren-mianshi"
}

# 创建 RabbitMQ 连接
def get_rabbitmq_connection():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    return pika.BlockingConnection(parameters)

def parse_task_payload(body):
    """
    按 Java 端固定结构解析：
    {"data":"...","type":"BQFK","userId":"...","taskId":"...","timestamp":...}
    """
    task_json = body.decode('utf-8')
    task_data = json.loads(task_json)
    return {
        "data": task_data.get("data", ""),
        "type": task_data.get("type"),
        "userId": task_data.get("userId"),
        "taskId": task_data.get("taskId"),
        "timestamp": task_data.get("timestamp"),
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
        print(f"[BQFK] Dify API 响应: {response}")
        if response.status_code == 200:
            result = response.json()
            return json.loads(result['answer'])
    except Exception as e:
        print(f"[BQFK ERROR] Dify API 请求失败: {e}")
    return None

def extract_and_analyze(task_id, user_id, element, interval_seconds=60):
    """视频抽帧并分析"""
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    
    # element 格式: videoUrl___msId___index
    data = element.split("___")
    video_filename = data[0]
    video_dir = os.path.splitext(video_filename)[0]
    ms_id = data[1]
    index = data[2]

    cap = cv2.VideoCapture(BASE_URL + video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"[BQFK ERROR] 视频无法打开: {BASE_URL + video_filename}")
        return

    print(f"视频帧数: {fps}")
    # 每 interval_seconds 秒抽取一帧（默认 60 秒）
    frame_interval = max(1, int(fps * interval_seconds))
    frame_count = 0
    folder_path = os.path.join(FRAME_DIR, video_dir)
    os.makedirs(folder_path, exist_ok=True)

    final_result = {"status": "NO_HUMAN_DETECTED"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rel_filename = f"{video_dir}/{frame_count}.jpg"
            frame_filename = os.path.join(FRAME_DIR, rel_filename)
            print(f"[BQFK] 抽帧保存路径: {frame_filename}")
            ok = cv2.imwrite(frame_filename, frame)
            if not ok:
                print(f"[BQFK ERROR] 抽帧保存失败: {frame_filename}")
                frame_count += 1
                continue
            
            image_url = REMOTE_BASE_URL + rel_filename
            print(f"[BQFK] Dify 图片地址: {image_url}")
            res = analyze_image(image_url)
            print(f"[BQFK] 抽帧分析结果: {res}")
            
            # 仅在检测到人物时入库
            summary = str((res or {}).get("summary", ""))
            no_human = ("未检测到人物" in summary)
            if res and not no_human:
                sql = "INSERT INTO b_mianshi_bq (ms_id, `index`, content, sort, url) VALUES (%s, %s, %s, %s, %s)"
                values = (
                    ms_id,
                    index,
                    summary,
                    str((res or {}).get("score", "")),
                    rel_filename
                )
                cursor.execute(sql, values)
                connection.commit()
                final_result = res
                # 抽整个视频都分析完，再入库
                # break
        frame_count += 1

    cap.release()
    cursor.close()
    connection.close()

    # 通过 RabbitMQ 发布通知 (Java 端会静默入库)
    notify_data = {
        "userId": user_id,
        "taskId": task_id,
        "type": "BQFK",
        "result": final_result
    }
    
    # 发布到 RabbitMQ Exchange
    notify_connection = get_rabbitmq_connection()
    notify_channel = notify_connection.channel()
    notify_channel.exchange_declare(exchange=NOTIFY_EXCHANGE, exchange_type='topic', durable=True)
    notify_channel.basic_publish(
        exchange=NOTIFY_EXCHANGE,
        routing_key=NOTIFY_ROUTING_KEY,
        body=json.dumps(notify_data, ensure_ascii=False).encode('utf-8'),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type='application/json',
            content_encoding='utf-8'
        )  # 持久化 + UTF-8 编码
    )
    notify_connection.close()
    
    print(f"[BQFK] 任务 {task_id} 处理完毕")

print(f"[BQFK Worker] 启动，正在监听队列: {QUEUE_BQFK}")

while True:
    try:
        # 创建 RabbitMQ 连接和通道
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # 声明队列
        channel.queue_declare(queue=QUEUE_BQFK, durable=True)
        
        # 阻塞消费消息
        method_frame, header_frame, body = channel.basic_get(queue=QUEUE_BQFK, auto_ack=False)
        
        if method_frame is None:
            connection.close()
            import time
            time.sleep(1)
            continue
            
        # 解析任务数据
        task_data = parse_task_payload(body)
        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        element = task_data.get("data") # videoUrl___msId___index
        if not isinstance(element, str):
            element = str(element)

        print(f"[BQFK] 收到任务 {task_id}, 用户 {user_id}")
        extract_and_analyze(task_id, user_id, element)
        
        # 确认消息已处理
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        connection.close()

    except Exception as e:
        print(f"[BQFK ERROR] 循环异常: {e}")
        import traceback
        traceback.print_exc()
        import time
        time.sleep(1)
