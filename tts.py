from f5_tts.api import F5TTS
import redis

f5tts = F5TTS(ckpt_file="F5-TTS/ckpts/model_1200000.safetensors",
        vocab_file="F5-TTS/ckpts/vocab.txt")
r = redis.Redis(host='localhost', port=6379, password='123456')

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

    print(f"生成文件: {filename}")
    
    #wav, sr, spect = f5tts.infer(
    #    ref_file="data/video/output_audio.wav",
    #    ref_text="各位朋友，当朝阳穿透薄雾，照亮新一天的征程，当我们怀揣梦想，踏上创业的冒险之旅。",
    #    #ref_text="让我们一起逐梦创业路，无畏前行，创就非凡。",
    #    gen_text=msg,
    #    file_wave=filename,
    #    seed=-1,  # random seed = -1
    #)
    wav, sr, spect = f5tts.infer(
        ref_file="output_audio.wav",
        ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
        gen_text=msg,
        file_wave=filename,
        seed=-1,  # random seed = -1
    )
    r.set(key, 1)

