import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack,AudioStreamTrack
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType
from av import AudioFrame, VideoFrame
import time
import fractions
import soundfile as sf
import resampy
import numpy as np
import pickle
import aiohttp
import redis

# user_id = 'sang'
r = redis.Redis(host='localhost', port=6379, password=None)
# video_num = int(r.get(user_id + '_all'))
# frames = []
# for i in range(video_num - 1):
#     frame = cv2.imread(f'results/avatars/avator_6/tmp/{str(i)}.png')
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames.append(np.array(frame))
    



# 自定义视频轨道类，从摄像头获取视频帧并提供给WebRTC
class VideoStreamTrack1(VideoStreamTrack):
    """
    一个自定义的视频流轨道，用于从摄像头读取视频帧并传递给WebRTC进行传输。
    """
    def __init__(self, zbjname, audio_track):
        super().__init__()
        self.zbjname = zbjname
        self.img_index = 0
        self.video_num = 0
        self.audio_name = r.get(zbjname)
        if self.audio_name:
            self.audio_name = self.audio_name.decode('utf-8')

        image = cv2.imread("0.png")
        image = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
        self.temp_frames = np.array(image)
        self.temp_frames2 = self.temp_frames
        self.bofang = False
        self.audio_track = audio_track

    async def next_timestamp(self) :
        if hasattr(self, "_timestamp"):
            self._timestamp += int(1 / 25 * 90000)
            wait = self._start + (self._timestamp / 90000) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, fractions.Fraction(1, 90000)
    
    async def recv(self):
        """
        重写recv方法，不断从摄像头读取视频帧，封装为帧对象返回。
        """
        try:
            frame = self.temp_frames
            if self.audio_name is None:
                frame = self.temp_frames
                # print('2', r.get(self.zbjname))
                #再次获取
                temp_audio_name = r.get(self.zbjname)
                if temp_audio_name:
                    temp_audio_name = temp_audio_name.decode('utf-8')
                if temp_audio_name and temp_audio_name != self.audio_name:
                    self.audio_name = temp_audio_name #再次获取
                    # print('3', self.audio_name, str(r.get(self.audio_name)), "*********************")
                    self.video_num = int(r.get(self.audio_name))
                else:
                    self.video_num = 0
            else:
                # zbjname 为0无预测，为1有预测
                if self.img_index >= self.video_num - 1:
                    # self.audio_name = str(r.get(zbjname))
                    self.img_index = 0
                    self.bofang = False
                    frame = self.temp_frames

                    #再次获取
                    temp_audio_name = r.get(self.zbjname)
                    if temp_audio_name:
                        temp_audio_name = temp_audio_name.decode('utf-8')
                    if temp_audio_name and temp_audio_name != self.audio_name:
                        self.audio_name = temp_audio_name #再次获取
                        # print('3', self.audio_name, str(r.get(self.audio_name)))
                        self.video_num = int(r.get(self.audio_name))
                    else:
                        self.video_num = 0
                else:
                    
                    # key = self.audio_name + str(self.img_index)
                    # while r.exists(key) is False:  #直到key存在
                    #     time.sleep(1)
                    # print(123123)
                    # await asyncio.sleep(0.1)
                    
                    # 不会掉帧
                    if self.audio_track.bofang:
                        self.bofang = True
                    if self.bofang:
                        key = self.audio_name + str(self.img_index)
                        val = r.get(key)
                        if val:
                            print(key, "********")
                            r.delete(key)
                            frame = pickle.loads(val)
                            self.temp_frames2 = frame
                        else:
                            frame = self.temp_frames2
                        
                        self.img_index += 1

                    # # 会掉帧
                    # self.img_index = int(self.audio_track.current_frame / 320 / 2)
                    # key = self.audio_name + str(self.img_index)
                    
                    # val = r.get(key)
                    # if val:
                    #     print(key, "********")
                    #     self.temp_frames2 = frame
                    #     r.delete(key)
                    #     frame = pickle.loads(val)
                    # else:
                    #     frame = self.temp_frames2
                        # ----------------------
                    # key2 = self.audio_name + str(100) #阈值
                    # if r.exists(key) and r.exists(key2):
                        
                    #     frame = pickle.loads(r.get(key))
                    #     r.delete(key)
                    #     self.temp_frames2 = frame
                    #     self.img_index += 1
                    # else:
                    #     frame = self.temp_frames2
                    
                    # self.img_index += 1
                                        

            frame = VideoFrame.from_ndarray(frame, format="bgr24")
            
            pts, time_base = await self.next_timestamp()
            # 填充视频帧参数
            frame.pts = pts
            frame.time_base = time_base
            
            # 返回视频帧
            return frame

        except IOError as e:
            print(f"摄像头读取出现异常，异常信息为: {e}")
            # 这里可以添加一些尝试重新初始化摄像头等操作的逻辑
        except Exception as e:
            print(f"其他未知异常导致摄像头视频帧读取失败，异常信息为: {e}")

# 自定义视频轨道类，从摄像头获取视频帧并提供给WebRTC
class AudioStreamTrack1(AudioStreamTrack):
    """
    一个自定义的音频流轨道，用于从摄像头读取视频帧并传递给WebRTC进行传输。
    """
    def __init__(self, zbjname):
        super().__init__()
        self.zbjname = zbjname
        fps = 50 # 20 ms per frame
        self.sample_rate = 16000
        self.frame_size = int(self.sample_rate // fps)  #注意
        # self.video_track = video_track
        # self.stream, self.sample_rate = sf.read("output_audio.wav")
        # print(f'[INFO]tts audio stream {self.sample_rate}: {self.stream.shape}')
        # self.stream = self.stream.astype(np.float32)

        # if self.stream.ndim > 1:
        #     print(f'[WARN] audio has {self.stream.shape[1]} channels, only use the first.')
        #     self.stream = self.stream[:, 0]

        # if self.sample_rate != sample_rate_flag and self.stream.shape[0]>0:
        #     print(f'[WARN] audio sample rate is {self.sample_rate}, resampling into {sample_rate_flag}.')
        #     self.stream = resampy.resample(x=self.stream, sr_orig=self.sample_rate, sr_new=sample_rate_flag)
        
        # self.sample_rate = sample_rate_flag
        # self.frame_size = int(chunk)
        self.stream = None
        self.total_frames = 0
        self.video_num = 0
        self.audio_name = r.get(zbjname)
        if self.audio_name:
            self.audio_name = self.audio_name.decode('utf-8')
        # if self.audio_name is not None:
        #     # self.stream = int(r.get(self.audio_name))
        #     self.stream = pickle.loads(r.get(self.audio_name + "_audio"))
        #     self.total_frames = self.stream.shape[0]
        self.current_frame = 0
        self.bofang = False
        
        
    async def recv(self):
        try:
            # await asyncio.sleep(0.1)
            # samples = int(AUDIO_PTIME * sample_rate)
            # if self.current_frame >= self.total_frames:
            #     # 如果已经读到文件末尾，重新回到文件开头
            #     # self.wav_file.rewind()
            #     self.current_frame = 0
            # print(self.video_track.img_index)
            # self.current_frame = self.video_track.img_index * self.frame_size
        
            flag_data = False
            row_data= None
            if self.audio_name is None:

                #再次获取
                temp_audio_name = r.get(self.zbjname)
                if temp_audio_name:
                    temp_audio_name = temp_audio_name.decode('utf-8')
                # print(temp_audio_name, "***************")
                if temp_audio_name and temp_audio_name != self.audio_name:
                    self.audio_name = temp_audio_name #再次获取
                    # print(temp_audio_name, "**********&&&&&&")
                    self.stream = pickle.loads(r.get(self.audio_name + "_audio"))
                    r.delete(self.audio_name + "_audio")
                    self.total_frames = self.stream.shape[0]
                    self.video_num = int(r.get(self.audio_name))
                else:
                    self.total_frames = 0
            else:
                # zbjname
                if self.current_frame >= self.total_frames:
                    # self.audio_name = str(r.get(zbjname))
                    self.current_frame = 0
                    self.bofang = False
                    # frame = self.temp_frames

                    #再次获取
                    temp_audio_name = r.get(self.zbjname)
                    if temp_audio_name:
                        temp_audio_name = temp_audio_name.decode('utf-8')
                    if temp_audio_name and temp_audio_name != self.audio_name:
                        self.audio_name = temp_audio_name #再次获取
                        self.stream = pickle.loads(r.get(self.audio_name + "_audio"))
                        r.delete(self.audio_name + "_audio")
                        self.total_frames = self.stream.shape[0]
                        self.video_num = int(r.get(self.audio_name))
                    else:
                        self.total_frames = 0
                else:
                    if self.bofang:
                        flag_data = True
                        row_data = self.stream[self.current_frame:self.current_frame+self.frame_size]
                        # print(self.current_frame, self.current_frame+self.frame_size)
                        # print(len(self.stream))
                        if self.current_frame+self.frame_size > len(self.stream):
                            flag_data = False
                        self.current_frame += self.frame_size

            if hasattr(self, "_timestamp"):
                self._timestamp += self.frame_size
                wait = self._start + (self._timestamp / self.sample_rate) - time.time()
                await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
            
            # raw_data = self.stream[self.current_frame:self.current_frame+self.frame_size]
            frame = AudioFrame(format="s16", layout="mono", samples=self.frame_size)
            # self.current_frame += self.frame_size
            # and self.video_track.bofang
            if flag_data and self.bofang:
                # print("jinlaile$$$$$$$$$$$$$")
                row_data = (row_data * 32767).astype(np.int16)
                frame.planes[0].update(row_data.tobytes())
                
            else:
                silent_bytes = bytes([0x00] * (self.frame_size * 2))  # s16类型每个采样占2字节
                frame.planes[0].update(silent_bytes)
                
                if self.audio_name and r.exists(self.audio_name + str(int(self.video_num // 1.2))):
                    # print(self.audio_name + str(240))
                    self.bofang = True
                # print("按照特定格式处理为静音数据")
                # frame.planes[0].update(None)
            #     print("$$$$$$$$$$$$$$$$$$$")
                # return AudioFrame(format="s16", layout="mono", samples=self.frame_size)

            frame.pts = self._timestamp
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame
        except Exception as e:
            print(f"其他未知异常导致帧读取失败，异常信息为: {e}")

class HumanSRS:
    def __init__(self, zbjname, zbjip):
        self.zbjname = zbjname
        self.stop = False
        self.zbjip = zbjip

    async def create_offer_and_send(self, pc, srs_whip_url):
        """
        创建WebRTC offer并发送到SRS服务器。

        :param pc: RTCPeerConnection实例
        :param srs_whip_url: SRS服务器的W-HIP URL
        """
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        
        async with aiohttp.ClientSession() as session:
            # 使用给定的WHIP地址
            async with session.post(srs_whip_url, data=pc.localDescription.sdp) as response:
                answer_data = await response.text()
                print(answer_data, "*****")
                answer = RTCSessionDescription(sdp=answer_data, type='answer')
                await pc.setRemoteDescription(answer)


    async def main(self):
        srs_whip_url = f"http://{self.zbjip}:1985/rtc/v1/whip/?app=live&stream={self.zbjname}&eip={self.zbjip}"
        pc = RTCPeerConnection()
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(srs_whip_url)
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()

        audio_track = AudioStreamTrack1(self.zbjname)

        video_track = VideoStreamTrack1(self.zbjname, audio_track)
        
        pc.addTrack(audio_track)
        pc.addTrack(video_track)

        await self.create_offer_and_send(pc, srs_whip_url)

        # # 保持程序运行，持续推送视频流，这里简单使用一个循环，可以根据实际需求改进退出机制等
        # while True:
        #     if self.stop:
        #         pc.close()
        #         break
        #     await asyncio.sleep(1)

    def run(self):
        # asyncio.run(self.main())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main())
        loop.run_forever() 
        


# if __name__ == "__main__":
#     # asyncio.run(main())
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(main())
#     loop.run_forever()  