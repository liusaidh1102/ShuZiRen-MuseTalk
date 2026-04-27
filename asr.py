import pika
import os
import json
from funasr import AutoModel

# RabbitMQ 配置与规范
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'admin'
RABBITMQ_PASS = '123456'
RABBITMQ_VHOST = 'ai-mianshi'
QUEUE_ASR = 'interview.queue.asr'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
NOTIFY_EXCHANGE = "interview.notify.exchange"
NOTIFY_ROUTING_KEY = "notify.result"
ASR_INPUT_DIR = os.path.join("tests", "asr")

# 初始化 ASR 模型
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  device="cuda",
                  )

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
    {"data":"...","type":"ASR","userId":"...","taskId":"...","timestamp":...}
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

print(f"[ASR Worker] 启动，正在监听队列: {QUEUE_ASR}")

while True:
    try:
        # 创建 RabbitMQ 连接和通道
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # 声明队列
        channel.queue_declare(queue=QUEUE_ASR, durable=True)
        
        # 阻塞消费消息
        method_frame, header_frame, body = channel.basic_get(queue=QUEUE_ASR, auto_ack=False)
        
        if method_frame is None:
            connection.close()
            import time
            time.sleep(1)
            continue

        # 解析任务数据 (JSON 格式)
        try:
            task_data = parse_task_payload(body)
        except json.JSONDecodeError as e:
            print(f"[ASR ERROR] JSON解析失败: {e}, 原始数据: {body.decode('utf-8')}")
            continue

        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        data = task_data.get("data") # 音频文件名
        if not isinstance(data, str):
            data = str(data)

        print(f"[ASR] 收到任务 {task_id}, 用户 {user_id}, 文件 {data}")

        # 准备处理：统一从 tests/asr 目录读取
        filename = os.path.basename(data)
        file_path = os.path.join(ASR_INPUT_DIR, filename)

        result_text = ""
        try:
            if os.path.exists(file_path):
                res = model.generate(input=file_path, batch_size_s=300, hotword='魔搭')
                result_text = res[0]["text"]
            else:
                print(f"[ASR ERROR] 文件不存在: {file_path}")
                result_text = "ERROR: FILE_NOT_FOUND"
        except Exception as e:
            print(f"[ASR ERROR] 模型处理失败: {e}")
            result_text = f"ERROR: {str(e)}"

        # 1. 更新任务状态为 DONE
        # 注意：这里仍使用 Redis 存储状态和结果，仅通知改用 RabbitMQ
        import redis
        r = redis.Redis(host='localhost', port=6379, password='123456', decode_responses=True)
        r.setex(TASK_STATUS.format(taskId=task_id), 600 * 180, "DONE")

        # 2. 存储结果到 Redis
        result_content = {"text": result_text}
        r.setex(TASK_RESULT.format(taskId=task_id), 600 * 180, json.dumps(result_content))

        # 3. 通过 RabbitMQ 发布 Pub/Sub 通知给 Java
        notify_data = {
            "userId": user_id,
            "taskId": task_id,
            "type": "ASR",
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

        # 确认消息已处理
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        connection.close()

        print(f"[ASR] 任务 {task_id} 处理完毕: {result_text}")

    except Exception as e:
        print(f"[ASR ERROR] 循环异常: {e}")
        import traceback
        traceback.print_exc()
        import time
        time.sleep(1)