from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import uuid
import redis
import threading
import pickle
import soundfile as sf
import resampy
import numpy as np
import time
import requests
import json
app = Flask(__name__)
CORS(app, resources={r"/tests/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
r = redis.Redis(host='localhost', port=6379, password='123456')
if not os.path.exists("tests"):
    os.makedirs("tests")

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get('Origin')
    if origin:
        resp.headers['Access-Control-Allow-Origin'] = origin
    else:
        resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,HEAD,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = request.headers.get('Access-Control-Request-Headers', '*')
    resp.headers['Access-Control-Expose-Headers'] = 'Content-Range, Range'
    resp.headers['Vary'] = 'Origin'
    return resp

@app.route('/tests/<path:filename>', methods=['GET','HEAD','OPTIONS'])
def serve_tests(filename):
    if request.method == 'OPTIONS':
        return ('', 204)
    if filename.lower().endswith('.wav'):
        print(f"[WAV] {request.method} {request.url} -> tests/{filename}")
    return send_from_directory('tests', filename)

app.config['ttsflag'] = 1
@app.route('/tts')
def index1():
    msg = request.args.get('msg')
    print(msg)
    if msg == None or msg == "":
        return jsonify({'message': ""}), 200
    key = str(uuid.uuid4())
    # 文字转语音，并将语音存到test文件夹下
    filename = "tests/" + str(key) + ".wav"
    r.rpush("ttsqueue" + str(app.config['ttsflag']), str(key) + "___" + str(msg))
    app.config['ttsflag'] = app.config['ttsflag'] + 1
    if app.config['ttsflag'] > 1:
        app.config['ttsflag'] = 1
    flag = 0
    while not r.exists(key):
        if flag > 20:
            break
        flag = flag + 1
        time.sleep(0.5)

    r.delete(key)
    #wav, sr, spect = f5tts.infer(
    #    ref_file="output_audio.wav",
    #    ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
    #    gen_text=msg,
    #    file_wave=filename,
    #    seed=-1,  # random seed = -1
    #)
    #print("seed :", f5tts.seed)
    return jsonify({'message': filename}), 200

@app.route('/asr')
def asr():
    msg = request.args.get('url')
    msg = msg.replace("tests/", "")
    print(msg)
    if msg == None or msg == "":
        return jsonify({'message': ""}), 200
    
    r.rpush("asrqueue1", str(msg))
    flag = 0
    while not r.exists(msg):
        if flag > 20:
            break
        flag = flag + 1
        time.sleep(0.5)
    text = r.get(msg).decode('utf-8')
    r.delete(msg)
    return jsonify({'message': text}), 200

@app.route('/bqfk')
def bqfk():
    msg = request.args.get('url') + "___" + request.args.get('msId') + "___" + request.args.get('index')
    print(msg)
    if msg == None or msg == "":
        return jsonify({'message': ""}), 200

    r.rpush("bqfkqueue1", str(msg))
    return jsonify({'message': 'success'}), 200

def gen_question(job, count):
    headers = {
        # 通过简历生成面试题apikey
        "Authorization": f"Bearer app-aAhdvh4iUdV39pj5e6g0QFfY",
        "Content-Type": "application/json"
    }
    # 这里需要根据 Dify API 文档调整请求体
    payload = {
        "inputs": {"job":job,"count": count},
        "query": "开始",
        "response_mode": "blocking",
        "user": "python-gen-question",
    }
    response = requests.post("http://localhost:8001/v1/chat-messages", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return json.loads(result['answer'])['question']

    else:
        print(f"API 请求失败，状态码: {response.status_code}，错误信息: {response.text}")



def audioHandle(zbjname, audio_url):
    audio_name = zbjname + "-" + audio_url
    stream_name = audio_name + "_audio"
    audio_chunk(audio_url, stream_name)

    whisper_feature = audio_processor.audio2feat(audio_url)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=25)

    print(len(whisper_chunks))

    # r.set(audio_name, len(whisper_chunks)) #总帧数
    r.psetex(audio_name, 3000000, len(whisper_chunks)) #过期时间50分钟

    # print(audio_name, len(whisper_chunks), "**********")
    for i, w in enumerate(whisper_chunks):
        serialized_w = pickle.dumps(w)
        # r.set(user_id + 'counter', 0)
        r.set(audio_name + "_" + str(i), serialized_w) #音频chunk
    print('audio_name', audio_name)

    # r.set(zbjname, audio_name) #播放语音url
    r.psetex(zbjname, 3000000, audio_name)
    
    r.rpush('infer_queue', audio_name)

def diaodu():
    queueList = []
    for i in range(5):
        queueList.append("queue" + str(i+1))
    #queueList = ["queue1", "queue2", "queue3", "queue4", "queue5", "queue6", "queue7", "queue8"]
    while True:
        # 阻塞式地从队列右边弹出一个元素（如果队列为空会一直等待，直到有元素可弹出）
        result = r.blpop('infer_queue', timeout=0)
        if result:
            # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
            audio_name = result[1].decode('utf-8')
            print(result)
            print(audio_name, "123213")
            video_num = int(r.get(audio_name))
            print(f"总帧数: {video_num}")
            # r.set(audio_name + "infer", 0)
            batch_size = 6
            flag = 0
            while flag <= video_num:
                for i in range(len(queueList)):
                    print(queueList[i], str(video_num) + "___" + str(flag) + "___" + str(audio_name))
                    r.rpush(queueList[i], str(video_num) + "___" + str(flag) + "___" + str(audio_name))
                    flag += batch_size
                
        else:
            print("队列为空，继续等待...")

def audio_chunk(audio_url, stream_name):
    fps = 50 # 20 ms per frame
    sample_rate_flag = 16000
    chunk = sample_rate_flag // fps

    stream, sample_rate = sf.read(audio_url)

    print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
    stream = stream.astype(np.float32)

    if stream.ndim > 1:
        print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
        stream = stream[:, 0]

    if sample_rate != sample_rate_flag and stream.shape[0]>0:
        print(f'[WARN] audio sample rate is {sample_rate}, resampling into {sample_rate_flag}.')
        stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=sample_rate_flag)

    r.set(stream_name, pickle.dumps(stream))


if __name__ == '__main__':
    thread = threading.Thread(target=diaodu)
    thread.start()
    app.run(debug=False,port=8181,host="0.0.0.0")


