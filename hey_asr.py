# 测试asr语音转写功能
from funasr import AutoModel
# 先pip安转 modelscope  funasr torch
# 加载一站式 ASR 模型：语音识别 + 静音检测 + 自动加标点
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc-c",
    device="cuda"  # 有 GPU 就改成 cuda
)

# 音频文件路径（支持 wav、mp3、m4a 等）
audio_file = "./tests/72a81d93-e2f2-49e0-b774-1a08d3dc6516.wav"

# 语音识别
result = model.generate(
    input=audio_file,
    hotword="自定义关键词",  # 热词增强，提高专业词识别率
    batch_size_s=300
)

# 输出结果
text = result[0]["text"]
print("识别结果：", text)