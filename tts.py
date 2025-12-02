# ===== 1. 先打补丁：替换 torchaudio.load 为 soundfile 实现 =====
import torchaudio
import soundfile
import torch
import numpy as np

def patched_torchaudio_load(filepath, *args, **kwargs):
    """
    用 soundfile 替代 torchaudio.load，避免 torchcodec 依赖。
    返回格式: (Tensor[float32] of shape [channels, time], sample_rate)
    """
    data, sr = soundfile.read(filepath, dtype='float32')
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # (1, T)
    else:
        data = data.T  # (C, T)
    return torch.from_numpy(data), sr

# 替换原始函数
torchaudio.load = patched_torchaudio_load
# ============================================================

# ===== 2. 现在再导入其他模块 =====
from f5_tts.api import F5TTS
import redis

f5tts = F5TTS(
    ckpt_file="F5-TTS/ckpts/model_1200000.safetensors",
    vocab_file="F5-TTS/ckpts/vocab.txt"
)
r = redis.Redis(host='localhost', port=6379, password='123456')

# ===== 3. 主循环保持不变 =====
while True:
    queue_result = r.blpop('ttsqueue1', timeout=0)
    element = ""
    if queue_result:
        element = str(queue_result[1].decode('utf-8')).split("___")
        key = element[0]
        msg = element[1]
        print(f"从队列中取出元素: {element}")
    else:
        print("队列为空，继续等待...")
        continue

    filename = "tests/" + key + ".wav"
    print(f"生成文件: {filename}")

    wav, sr, spect = f5tts.infer(
        ref_file="output_audio.wav",
        ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
        gen_text=msg,
        file_wave=filename,
        seed=-1,
    )
    r.set(key, 1)