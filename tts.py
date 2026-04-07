import redis
import torch
import json
import hashlib
import os
from f5_tts.api import F5TTS

# Redis 配置与规范
r = redis.Redis(host='localhost', port=6379, password='123456', decode_responses=True)
QUEUE_TTS = 'interview:queue:tts'
TASK_STATUS = "interview:task:{taskId}:status"
TASK_RESULT = "interview:task:{taskId}:result"
CACHE_TTS = "interview:cache:tts:{md5}"
CHANNEL_NOTIFY = "interview:channel:notify"
BASE_URL = "http://localhost:4000/tests/" # 假设音频通过 Nginx 暴露

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

while True:
    try:
        # BLPOP 阻塞监听队列
        queue_result = r.blpop(QUEUE_TTS, timeout=0)
        if not queue_result:
            continue

        # 解析任务数据 (JSON 格式)
        task_json = queue_result[1]
        
        # 兼容处理：如果 Java 端传过来的是带引号的字符串，先去掉引号
        if task_json.startswith('"') and task_json.endswith('"'):
            task_json = json.loads(task_json) # 第一次 loads 去掉外层字符串转义
            
        task_data = json.loads(task_json)
        task_id = task_data.get("taskId")
        user_id = task_data.get("userId")
        msg = task_data.get("data") # 合成文本

        print(f"[TTS] 收到任务 {task_id}, 用户 {user_id}, 文本: {msg[:20]}...")

        # 生成音频文件名
        filename = f"{task_id}.wav"
        save_path = os.path.join("tests", filename)

        # 执行 TTS 推理
        try:
            wav, sr, spect = f5tts.infer(
                ref_file="output_audio.wav",
                ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
                gen_text=msg,
                file_wave=save_path,
                seed=-1,
            )
            audio_url = BASE_URL + filename
            print(f"[TTS] 合成成功: {save_path}")
        except Exception as e:
            print(f"[TTS ERROR] 推理失败: {e}")
            audio_url = f"ERROR: {str(e)}"

        # 1. 更新任务状态为 DONE
        r.setex(TASK_STATUS.format(taskId=task_id), 600, "DONE")

        # 2. 存储结果到 Redis
        result_content = {"audioUrl": audio_url}
        # 过期时间60s
        r.setex(TASK_RESULT.format(taskId=task_id), 600, json.dumps(result_content))

        # 3. 设置 TTS 缓存 (基于 MD5)
        md5 = hashlib.md5(msg.encode()).hexdigest()
        r.setex(CACHE_TTS.format(md5=md5), 3600, audio_url)

        # 4. 发布通知给 Java
        notify_data = {
            "userId": user_id,
            "taskId": task_id,
            "type": "TTS",
            "result": result_content
        }
        r.publish(CHANNEL_NOTIFY, json.dumps(notify_data))

        print(f"[TTS] 任务 {task_id} 处理完毕")

    except Exception as e:
        print(f"[TTS ERROR] 循环异常: {e}")
        import time
        time.sleep(1)
