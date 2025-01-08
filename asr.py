import redis
from paddlespeech.cli.asr.infer import ASRExecutor
asr = ASRExecutor()
r = redis.Redis(host='10.23.32.63', port=6389, password=None)

while True:
    queue_result = r.blpop('asrqueue1', timeout=0)
    element = ""
    if queue_result:
         # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
        element = str(queue_result[1].decode('utf-8'))
        print(f"从队列中取出元素: {element}")
    else:
        print("队列为空，继续等待...")
        continue
    print('/project/resume/dist/apps/server/audios/' + element)
    result = ""
    try:
        result = asr(model='conformer_talcs',codeswitch=True,
        force_yes=False,lang='zh_en',audio_file='/project/resume/dist/apps/server/audios/' + element)
    except:
        print("错误")
    print(result)
    r.set(element, result)

