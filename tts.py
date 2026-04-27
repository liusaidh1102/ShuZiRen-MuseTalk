import pika
import torch
import json
import hashlib
import os
import redis
from f5_tts.api import F5TTS

# RabbitMQ 配置与规范
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'admin'
RABBITMQ_PASS = '123456'
RABBITMQ_VHOST = 'ai-mianshi'
QUEUE_TTS = 'interview.queue.tts'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
CACHE_TTS = "interview:cache:tts:{md5}"
TASK_PROCESSING = "interview:task:{taskId}:processing"
NOTIFY_EXCHANGE = "interview.notify.exchange"
NOTIFY_ROUTING_KEY = "notify.result"
TTS_OUTPUT_DIR = os.path.join("tests", "tts")
TTS_BASE_URL = "http://localhost:5000/tests/tts/"  # 返回给前端的 TTS 地址

# 初始化 F5-TTS 模型
print("正在加载 F5-TTS 模型...")
f5tts = F5TTS(
    ckpt_file="F5-TTS/ckpts/model_1200000.safetensors",
    vocab_file="F5-TTS/ckpts/vocab.txt",
    local_path="F5-TTS/ckpts"
)

print("是否使用 GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 型号:", torch.cuda.get_device_name(0))

print(f"[TTS Worker] 启动，正在监听队列: {QUEUE_TTS}")

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
    {"data":"...","type":"TTS","userId":"...","taskId":"...","timestamp":...}
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

while True:
    try:
        # 创建 RabbitMQ 连接和通道
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # 声明队列
        channel.queue_declare(queue=QUEUE_TTS, durable=True)
        
        # 阻塞消费消息
        method_frame, header_frame, body = channel.basic_get(queue=QUEUE_TTS, auto_ack=False)
        
        if method_frame is None:
            connection.close()
            import time
            time.sleep(1)
            continue

        # 解析任务数据 (JSON 格式)
        task_data = parse_task_payload(body)
        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        msg = task_data.get("data") # 合成文本
        if not isinstance(msg, str):
            msg = json.dumps(msg, ensure_ascii=False)
        if not task_id:
            print("[TTS ERROR] 缺少 taskId，丢弃消息")
            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            connection.close()
            continue

        r = redis.Redis(host='localhost', port=6379, password='123456', decode_responses=True)
        processing_key = TASK_PROCESSING.format(taskId=task_id)
        status_key = TASK_STATUS.format(taskId=task_id)
        result_key = TASK_RESULT.format(taskId=task_id)

        # 幂等处理：已完成任务直接 ACK，避免重复消费
        if r.get(status_key) == "DONE" and r.exists(result_key):
            print(f"[TTS] 任务 {task_id} 已完成，跳过重复消息")
            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            connection.close()
            continue

        # 任务互斥锁：同一 taskId 同时只能有一个消费者处理
        if not r.set(processing_key, "1", nx=True, ex=600 * 30):
            print(f"[TTS] 任务 {task_id} 正在处理中，跳过重复消息")
            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            connection.close()
            continue

        print(f"[TTS] 收到任务 {task_id}, 用户 {user_id}, 文本: {msg[:20]}...")

        # 生成音频文件名
        filename = f"{task_id}.wav"
        os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(TTS_OUTPUT_DIR, filename)

        # 执行 TTS 推理
        try:
            wav, sr, spect = f5tts.infer(
                ref_file="output_audio.wav",
                ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
                gen_text=msg,
                file_wave=save_path,
                seed=-1,
            )
            audio_url = TTS_BASE_URL + filename
            print(f"[TTS] 合成成功: {save_path}")
        except Exception as e:
            print(f"[TTS ERROR] 推理失败: {e}")
            audio_url = f"ERROR: {str(e)}"

        # 1. 更新任务状态为 DONE
        r.setex(TASK_STATUS.format(taskId=task_id), 600 * 180, "DONE")

        # 2. 存储结果到 Redis
        result_content = {"audioUrl": audio_url}
        # 过期时间180分钟
        r.setex(TASK_RESULT.format(taskId=task_id), 600 * 180, json.dumps(result_content))


        # 3. 通过 RabbitMQ 发布通知给 Java
        notify_data = {
            "userId": user_id,
            "taskId": task_id,
            "type": "TTS",
            "result": result_content
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
        r.delete(processing_key)
        
        # 确认消息已处理
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        connection.close()

        print(f"[TTS] 任务 {task_id} 处理完毕")

    except Exception as e:
        print(f"[TTS ERROR] 循环异常: {e}")
        try:
            if 'r' in locals() and 'processing_key' in locals():
                r.delete(processing_key)
        except Exception:
            pass
        import time
        time.sleep(1)
