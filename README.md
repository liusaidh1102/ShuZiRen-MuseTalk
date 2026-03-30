```markdown
# Human MS - 虚拟人唇形同步系统

基于 MuseTalk 和 F5-TTS 的实时高质量唇形同步虚拟人生成系统。

## 📖 项目简介

本项目是一个集成语音合成（TTS）、语音识别（ASR）和唇形同步的虚拟人生成系统，支持多语言音频驱动，可实时生成高质量的唇形同步视频（30fps+）。

### 核心功能

- **唇形同步**：基于潜空间修复技术的高质量唇形同步
- **语音合成**：集成 F5-TTS 文本转语音引擎
- **语音识别**：支持 ASR 语音转文字功能
- **实时推理**：支持正常推理与实时推理两种模式
- **多语言支持**：支持中英文等多语言音频驱动
- **嘴型调节**：可调节嘴型开合度（bbox_shift）

### 应用场景

- 虚拟人生成（配合 MuseV）
- 视频配音
- 直播互动
- 数字人内容创作

## 🏗️ 项目结构

```

human_ms/
├── F5-TTS/                 # F5-TTS 语音合成子模块
│   ├── src/f5_tts/        # F5-TTS 核心代码
│   │   ├── model/         # 模型定义
│   │   ├── infer/         # 推理脚本
│   │   ├── train/         # 训练脚本
│   │   └── api.py         # API 接口
│   └── ckpts/             # 预训练权重
├── data/                   # 数据目录
│   ├── audio/             # 音频资源
│   └── video/             # 视频资源
├── audios/                 # 音频输出目录
├── tests/                  # 测试文件目录
├── server.py              # Web 服务器主程序
├── tts.py                 # 异步 TTS 消费服务
├── asr.py                 # ASR 语音识别服务
├── bqfk.py                # 翻译服务
└── hey_tts.py             # TTS 调用示例
```
### 数据流向

- **输入素材**: `data/` 目录存放音频和视频输入素材
- **临时文件**: `tests/` 目录存放临时生成的文件
- **最终输出**: `results/` 目录存放输出的同步视频
- **资源库**: `audios/` 目录存放音频资源

## 🔧 环境要求

### 基本配置

- **Python**: >= 3.10
- **CUDA**: 11.7
- **GPU**: 推荐 NVIDIA V100 或更高性能显卡

### 外部依赖

- **Redis**: localhost:6379 (密码：123456)
- **FFmpeg**: 需设置 `FFMPEG_PATH` 环境变量

### 模型权重

系统会自动下载以下模型权重至 `./models` 目录：

- musetalk - 唇形同步模型
- whisper - 语音识别模型
- dwpose - 姿态估计模型
- face-parse - 人脸解析模型

## 🚀 快速开始

### 1. 安装依赖

```
bash
# 克隆项目
git clone <repository-url>
cd human_ms

# 安装 F5-TTS 子模块
cd F5-TTS
pip install -e .

# 返回根目录并安装主项目依赖
cd ..
pip install -r requirements.txt
```
### 2. 配置环境变量

设置 FFmpeg 路径（Windows 示例）：

```
powershell
$env:FFMPEG_PATH = "C:\path\to\ffmpeg"
```
### 3. 启动 Redis

确保 Redis 服务在本地运行：

```
bash
redis-server --requirepass 123456
```
### 4. 运行服务

#### 方式一：Web 服务

```
bash
python server.py
```
#### 方式二：TTS 服务

```
bash
python tts.py
```
#### 方式三：ASR 服务

```
bash
python asr.py
```
## 📝 使用说明

### 基本使用流程

1. **准备素材**：将视频和音频文件放入 `data/video/` 和 `data/audio/` 目录
2. **启动服务**：运行相应的服务脚本
3. **生成视频**：通过 API 或命令行工具生成唇形同步视频
4. **查看结果**：生成的视频保存在 `results/` 目录

### API 接口

服务启动后，可通过 HTTP API 进行调用：

```
bash
# TTS 文本转语音
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}'

# ASR 语音转文字
curl -X POST http://localhost:8000/asr \
  -F "audio=@audio.wav"

# 唇形同步视频生成
curl -X POST http://localhost:8000/generate \
  -F "video=@input.mp4" \
  -F "audio=@output_audio.wav"
```
### 参数调节

可通过 `bbox_shift` 参数调节嘴型开合度：

- 正值：增大嘴型开合度
- 负值：减小嘴型开合度
- 默认值：0

## 🛠️ 开发指南

### 核心模块

- **MuseTalk** (`musetalk/`): 唇形同步模型定义
- **推理脚本** (`scripts/`): 
  - `inference.py` - 标准推理
  - `realtime_inference.py` - 实时推理
- **F5-TTS** (`F5-TTS/src/f5_tts/`): 语音合成引擎

### 自定义模型

如需使用自定义模型权重，可将权重文件放入 `models/` 目录并在配置文件中指定路径。

### 性能优化

- 使用 GPU 推理可获得最佳性能
- 调整 batch_size 以平衡速度和质量
- 实时模式下可降低分辨率以提升帧率

## 📋 常见问题

### Q: 如何修改嘴型开合度？

A: 在调用推理脚本时传入 `bbox_shift` 参数：

```
python
bbox_shift = 5  # 增大嘴型开合度
```
### Q: 推理速度慢怎么办？

A: 尝试以下方法：
- 降低视频分辨率
- 使用实时推理模式
- 升级 GPU 硬件

### Q: 如何切换不同的 TTS 声音？

A: 修改 TTS 配置文件中的说话人 ID 或使用不同的参考音频。

## 📄 许可证

本项目基于相关开源协议发布，请查看各子项目的具体 LICENSE 文件。

## 🙏 致谢

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - 唇形同步基础模型
- [F5-TTS](https://github.com/SWivid/F5-TTS) - 语音合成引擎
- [Whisper](https://github.com/openai/whisper) - 语音识别

## 📞 联系方式

如有问题或建议，欢迎提交 Issue 或联系开发者。
```
