from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS, cross_origin
#from f5_tts.api import F5TTS
import uuid
#from srstl import HumanSRS
from hey_srs import HumanSRSv2
import redis
# import asyncio
import threading
# from musetalk.utils.utils import load_all_model
#from musetalk.whisper.audio2feature import Audio2Feature
import pickle
import soundfile as sf
import resampy
import numpy as np
from pydub import AudioSegment
import time
import requests
import json
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import shutil

app = Flask(__name__)
CORS(app,supports_credentials=True)
# asr = ASRExecutor()
#f5tts = F5TTS(ckpt_file="F5-TTS/ckpts/model_1200000.safetensors",
#        vocab_file="F5-TTS/ckpts/vocab.txt",local_path="F5-TTS/ckpts")
r = redis.Redis(host='localhost', port=6379, password='123456')
#audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")

# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists("tests"):
    os.makedirs("tests")

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

@app.route('/create/zbj')
def zbj():
    username = request.args.get('username')
    print(username)
    if username == None or username == "":
        username = "hnkjxy"
        #return jsonify({'message': ""}), 200
    zbjname = username + "*" + str(uuid.uuid4())
    print(zbjname)
    r.psetex(zbjname + "check", 360000, 1) #6分钟没有，自动关闭直播间
    
    #human = HumanSRS(zbjname, "srs.xiaozhu.com:2022")
    #thread = threading.Thread(target=human.run)
    #thread.start()

    return jsonify({'message': zbjname}), 200

@app.route('/create/zbjv2')
def zbjv2():
    username = request.args.get('username')
    job = request.args.get('job')
    count = request.args.get('count')
    print(username, job, count)
    if username == None or username == "":
        return jsonify({'message': ""}), 200
    zbjname = username + "*" + str(uuid.uuid4())
    print(zbjname)
    r.psetex(zbjname + "check", 360000, 1) #6分钟没有，自动关闭直播间

    #human = HumanSRSv2(zbjname, "srs.xiaozhu.com:2022")
    #thread = threading.Thread(target=human.run)
    #thread.start()

    question = gen_question(job, count)
    ttsflag = 0
    for que in question:
        r.rpush("tts_text", str(zbjname) + "___" + str(que) + "___" + str(ttsflag))
        ttsflag = ttsflag + 1
    print({'zbjname': zbjname, 'question': question})
    return jsonify({'zbjname': zbjname, 'question': question}), 200

def gen_question(job, count):
    headers = {
        "Authorization": f"Bearer app-kHDJw31r4XePPliSx993FvbW",
        "Content-Type": "application/json"
    }
    # 这里需要根据 Dify API 文档调整请求体
    payload = {
        "inputs": {"job":job,"count": count},
        "query": "开始",
        "response_mode": "blocking",
        "user": "python-gen-question",
    }
    response = requests.post("http://localhost/v1/chat-messages", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return json.loads(result['answer'])['question']

    else:
        print(f"API 请求失败，状态码: {response.status_code}，错误信息: {response.text}")


@app.route('/human')
def human():
    zbjname = request.args.get('zbjname')
    audio_url = request.args.get('audioUrl')
    if zbjname == None or zbjname == "":
        zbjname = 'hnkjxyms'
        #return jsonify({'message': "直播间不可为空"}), 200
    if audio_url == None or audio_url == "":
        return jsonify({'message': "音频不可为空"}), 200
    # print(zbjname)
    
    # human = HumanSRS(zbjname, "192.168.21.13")
    thread = threading.Thread(target=audioHandle, args=(zbjname, audio_url))
    thread.start()

    # 发送带有 JSON 数据的 POST 请求
    audio = AudioSegment.from_file(audio_url)
    # 获取时长（以毫秒为单位）
    duration_ms = len(audio)

    # 将时长转换为秒
    duration_seconds = duration_ms / 1000.0
   
   
    #while not r.exists(zbjname + "ok"):
    #    time.sleep(0.5)
    
    #r.delete(zbjname + "ok")

    return jsonify({'message': duration_seconds + 1.5}), 200

@app.route('/humanv2')
def humanv2():
    zbjname = request.args.get('zbjname')
    audio_url = request.args.get('audioUrl')
    if zbjname == None or zbjname == "":
        zbjname = 'hnkjxyms'
        #return jsonify({'message': "直播间不可为空"}), 200
    if audio_url == None or audio_url == "":
        return jsonify({'message': "音频不可为空"}), 200
    # 发送带有 JSON 数据的 POST 请求
    if str(audio_url) not in ["99", "98", "97","96"]:
        audio = AudioSegment.from_file('tests/' + zbjname + "_" + audio_url + ".wav")
        while not r.exists(zbjname + audio_url + "ok"):
            time.sleep(0.5)

        r.delete(zbjname + audio_url + "ok")
    else:
        shutil.copy2("tests/" +  audio_url + ".mp4", 'tests/' + zbjname + "_" + audio_url + ".mp4")
        audio = AudioSegment.from_file("tests/" + audio_url + ".wav")

    # 获取时长（以毫秒为单位）
    duration_ms = len(audio)

    # 将时长转换为秒
    duration_seconds = duration_ms / 1000.0
    
    r.psetex(zbjname + "_hey_flag", 3600000, audio_url)


    return jsonify({'message': duration_seconds + 1.5}), 200

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
    # streamlen = stream.shape[0]
    # print(streamlen)
    # idx = 0
    # flag = 0
    # while streamlen >= chunk:  #and self.state==State.RUNNING
    #     # self.put_audio_frame()
    #     stream_temp = stream[idx:idx+chunk]
    #     streamlen -= chunk
    #     idx += chunk
    #     # r.rpush('audio_queue', stream_temp.tobytes())
    #     serialized_w = pickle.dumps(stream_temp)
    #     # r.set(user_id + 'counter', 0)
    #     r.set(user_id + '_' + str(flag), serialized_w)

    #     flag += 1



# @app.route('/human')
# def human():
# # 欢迎使用虚拟面试，我是你的面试官，希望接下来的面试顺利进行，让我们共同努力！
#     path = request.args.get('path')
#     print(path)
#     if path == None or path == "":
#         return jsonify({'message': ""}), 200
#     url = 'http://localhost:8180/human'
#     json_data = {
#         'text': path,
#         'type': 'speech'
#     }
    
#     # 发送带有 JSON 数据的 POST 请求
#     response = requests.post(url, json=json_data)
#     return jsonify({'message': "ok"}), 200
 


if __name__ == '__main__':
    thread = threading.Thread(target=diaodu)
    thread.start()
    app.run(debug=False,port=8181,host="0.0.0.0")



# zbjname：username + "*" + str(uuid.uuid4())       key: 直播间名称，value: 语音key标识audio_name   通过直播间找到指定语音
# audio_name  = zbjname + "-" + audio_url           key: zbjname + 语音url，value: 语音总帧数
# stream_name  = audio_name + "_audio" key: audio_name + _audio，value: 推流语音数组
# zbjname + "check"                                  key:zbjname + check, value: 1   直播间存活判断
# audio_name + "_" + str(i)                        key: audio_name + _i, value: 预测语音chunk
# audio_name + str(flag)                           key: audio_name + i value: 预测图片
# infer_queue                                  key: infer_queue value:audio_name
# queue*                                       key: queue* value: 总帧数_开始预测帧数_audio_name

