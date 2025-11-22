import redis
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )
r = redis.Redis(host='localhost', port=6379, password='123456')

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
    element = element.replace("tests/", "")
    print('/home/user/project/human_ms/audios/' + element)
    result = ""
    try:
        res = model.generate(input="/home/user/project/human_ms/audios/" + element,
            batch_size_s=300,
            hotword='魔搭')
        result = res[0]["text"]
    except:
        print("错误")
    print(result)
    r.set(element, result)

