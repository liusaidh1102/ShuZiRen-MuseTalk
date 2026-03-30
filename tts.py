from f5_tts.api import F5TTS
import redis
import torch
'''
使用低版本的f5-tts，适配local_path参数
f5-tts深度学习模型
pip uninstall -y f5-tts
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --timeout 120 f5-tts==0.3.4
'''
f5tts = F5TTS(
    ckpt_file="F5-TTS/ckpts/model_1200000.safetensors",
    vocab_file="F5-TTS/ckpts/vocab.txt",
    local_path="F5-TTS/ckpts"
)
r = redis.Redis(host='localhost', port=6379, password='123456')


print("是否使用GPU:", torch.cuda.is_available())
print("GPU型号:", torch.cuda.get_device_name(0))

while True:
    queue_result = r.blpop('ttsqueue1', timeout=0)
    element = ""
    if queue_result:
         # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
        element = str(queue_result[1].decode('utf-8')).split("___")
        key = element[0]
        msg = element[1]
        print(f"从队列中取出元素: {element}")
    else:
        print("队列为空，继续等待...")
        continue
    filename = "tests/" + key + ".wav"
    wav, sr, spect = f5tts.infer(
        ref_file="output_audio.wav",
        ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
        gen_text=msg,
        file_wave=filename,
        # sample_rate=44100,
        # num_channels=2,
        seed=-1,  # random seed = -1
    )
    r.set(key, 1)