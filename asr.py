import redis
import os
import json
from funasr import AutoModel

# Redis 配置与规范
r = redis.Redis(host='localhost', port=6379, password='123456', decode_responses=True)
QUEUE_ASR = 'interview:queue:asr'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
CHANNEL_NOTIFY = "interview:channel:notify"
TESTS_DIR = r"tests"

# 初始化 ASR 模型
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  device="cuda",
                  )

print(f"[ASR Worker] 启动，正在监听队列: {QUEUE_ASR}")

while True:
    try:
        # BLPOP 阻塞监听队列
        queue_result = r.blpop(QUEUE_ASR, timeout=0)
        if not queue_result:
            continue

        # 解析任务数据 (JSON 格式)
        task_json = queue_result[1]

        # ====================== 修复位置：双重JSON解析 ======================
        try:
            task_data = json.loads(task_json)
            # 处理Java传来的双重转义JSON
            if isinstance(task_data, str):
                task_data = json.loads(task_data)
        except json.JSONDecodeError as e:
            print(f"[ASR ERROR] JSON解析失败: {e}, 原始数据: {task_json}")
            continue

        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        data = task_data.get("data") # 音频文件名

        print(f"[ASR] 收到任务 {task_id}, 用户 {user_id}, 文件 {data}")

        # 准备处理
        element = data.replace("tests/", "")
        file_path = os.path.join(TESTS_DIR, element)

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
        r.setex(TASK_STATUS.format(taskId=task_id), 600, "DONE")

        # 2. 存储结果到 Redis
        result_content = {"text": result_text}
        r.setex(TASK_RESULT.format(taskId=task_id), 600, json.dumps(result_content))

        # 3. 发布 Pub/Sub 通知给 Java
        notify_data = {
            "userId": user_id,
            "taskId": task_id,
            "type": "ASR",
            "result": result_content
        }
        r.publish(CHANNEL_NOTIFY, json.dumps(notify_data))

        print(f"[ASR] 任务 {task_id} 处理完毕: {result_text}")

    except Exception as e:
        print(f"[ASR ERROR] 循环异常: {e}")
        import time
        time.sleep(1)