import torch
import sys

print("=" * 60)
print("RTX 4060 GPU环境验证")
print("=" * 60)

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("🎉 RTX 4060 GPU配置成功！")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"\n🎮 GPU {i}: {gpu_props.name}")
        print(f"  显存: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"  计算能力: {gpu_props.major}.{gpu_props.minor}")
        print(f"  多处理器: {gpu_props.multi_processor_count}")
    
    # 性能测试
    print("\n🧪 性能测试...")
    
    # 测试1: 矩阵乘法
    size = 4096
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    
    print(f"✅ 矩阵乘法 ({size}x{size}): {elapsed_time:.2f} 毫秒")
    print(f"✅ 测试张量设备: {c.device}")
    
    # 内存测试
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"💾 显存使用: {allocated:.2f} GB / {reserved:.2f} GB")
    
    print("\n🚀 RTX 4060非常适合TTS任务！")
    print("预计性能:")
    print("  • TTS推理速度: 0.1-0.3秒/句")
    print("  • 支持批量处理")
    print("  • 实时语音合成")
    
else:
    print("❌ GPU配置失败")
    print("请检查:")
    print("1. 是否在tts_gpu环境中")
    print("2. PyTorch是否安装了CUDA 12.6版本")

print("=" * 60)