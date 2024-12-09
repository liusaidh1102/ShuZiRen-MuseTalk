from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS, cross_origin
from f5_tts.api import F5TTS
import uuid
import os
import sys
from srstl import HumanSRS
import redis
# import asyncio
import threading
# from musetalk.utils.utils import load_all_model
from musetalk.whisper.audio2feature import Audio2Feature
import pickle
import soundfile as sf
import resampy
import numpy as np
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
 
app = Flask(__name__)
CORS(app)

# asr = ASRExecutor()
f5tts = F5TTS()
r = redis.Redis(host='localhost', port=6379, password=None)
audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")

# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists("tests"):
    os.makedirs("tests")

@app.route('/tts')
def index1():
    msg = request.args.get('msg')
    print(msg)
    if msg == None or msg == "":
        return jsonify({'message': ""}), 200
    filename = "tests/" + str(uuid.uuid4()) + ".wav"
    wav, sr, spect = f5tts.infer(
        ref_file="output_audio.wav",
        ref_text="大家好，非常荣幸能够作为今天的面试官与各位见面。",
        gen_text=msg,
        file_wave=filename,
        seed=-1,  # random seed = -1
    )
    print("seed :", f5tts.seed)
    return jsonify({'message': filename}), 200

@app.route('/create/zbj')
def zbj():
    username = request.args.get('username')
    print(username)
    if username == None or username == "":
        return jsonify({'message': ""}), 200
    zbjname = username + "*" + str(uuid.uuid4())
    print(zbjname)
    
    human = HumanSRS(zbjname, "192.168.21.13")
    thread = threading.Thread(target=human.run)
    thread.start()

    return jsonify({'message': zbjname}), 200

@app.route('/human')
def human():
    zbjname = request.args.get('zbjname')
    audio_url = request.args.get('audioUrl')
    if zbjname == None or zbjname == "":
        return jsonify({'message': "直播间不可为空"}), 200
    if audio_url == None or audio_url == "":
        return jsonify({'message': "音频不可为空"}), 200
    # print(zbjname)
    
    # human = HumanSRS(zbjname, "192.168.21.13")
    # thread = threading.Thread(target=human.run)
    # thread.start()
    audio_name = zbjname + "-" + audio_url
    stream_name = audio_name + "_audio"
    audio_chunk(audio_url, stream_name)

    # self.audio_name + self.img_index

    

    whisper_feature = audio_processor.audio2feat(audio_url)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=25)

    print(len(whisper_chunks))

    whisper_batch = []
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

    return jsonify({'message': zbjname}), 200

def diaodu():
    queueList = ["queue1"]
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
            batch_size = 4
            flag = 0
            while flag <= video_num:
                for i in range(len(queueList)):
                    print(queueList[i], str(video_num) + "_" + str(flag) + "_" + str(audio_name))
                    r.rpush(queueList[i], str(video_num) + "_" + str(flag) + "_" + str(audio_name))
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
    app.run(debug=False,port=8181)