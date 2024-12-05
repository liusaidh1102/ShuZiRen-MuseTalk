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
import redis

user_id = 'sang'
r = redis.Redis(host='localhost', port=6379, password=None)
video_num = int(r.get(user_id + '_all'))
frames = []
for i in range(video_num - 1):
    frame = cv2.imread(f'results/avatars/avator_6/tmp/{str(i)}.png')
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(np.array(frame))
    



# 自定义视频轨道类，从摄像头获取视频帧并提供给WebRTC
class VideoStreamTrack1(VideoStreamTrack):
    """
    一个自定义的视频流轨道，用于从摄像头读取视频帧并传递给WebRTC进行传输。
    """
    def __init__(self):
        super().__init__()
        self.img_index = 0

        # self.cap = cv2.VideoCapture(0)  # 这里默认打开本地第一个摄像头，可根据需求修改参数

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
            
            if self.img_index >= video_num - 1:
                self.img_index = 0
            
            # frame = cv2.imread(f'results/avatars/avator_6/tmp/{str(self.img_index)}.png')
            # frame = np.array(frame)
            frame = frames[self.img_index]

            self.img_index += 1

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
    def __init__(self):
        super().__init__()
        fps = 50 # 20 ms per frame
        sample_rate_flag = 16000
        chunk = sample_rate_flag // fps   #注意
        self.stream, self.sample_rate = sf.read("output_audio.wav")
        print(f'[INFO]tts audio stream {self.sample_rate}: {self.stream.shape}')
        self.stream = self.stream.astype(np.float32)

        if self.stream.ndim > 1:
            print(f'[WARN] audio has {self.stream.shape[1]} channels, only use the first.')
            self.stream = self.stream[:, 0]

        if self.sample_rate != sample_rate_flag and self.stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {self.sample_rate}, resampling into {sample_rate_flag}.')
            self.stream = resampy.resample(x=self.stream, sr_orig=self.sample_rate, sr_new=sample_rate_flag)
        
        self.sample_rate = sample_rate_flag
        self.frame_size = int(chunk)
        self.total_frames = self.stream.shape[0]
        self.current_frame = 0
        print(self.sample_rate, self.frame_size)
        
        
    async def recv(self):
        try:
            # await asyncio.sleep(0.1)
            # samples = int(AUDIO_PTIME * sample_rate)
            if self.current_frame >= self.total_frames:
                # 如果已经读到文件末尾，重新回到文件开头
                # self.wav_file.rewind()
                self.current_frame = 0

            if hasattr(self, "_timestamp"):
                self._timestamp += self.frame_size
                wait = self._start + (self._timestamp / self.sample_rate) - time.time()
                await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
            
            raw_data = self.stream[self.current_frame:self.current_frame+self.frame_size]
            self.current_frame += self.frame_size

            raw_data = (raw_data * 32767).astype(np.int16)
            frame = AudioFrame(format="s16", layout="mono", samples=self.frame_size)
            frame.planes[0].update(raw_data.tobytes())
            frame.pts = self._timestamp
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame
        except Exception as e:
            print(f"其他未知异常导致帧读取失败，异常信息为: {e}")

async def create_offer_and_send(pc, srs_whip_url):
    """
    创建WebRTC offer并发送到SRS服务器。

    :param pc: RTCPeerConnection实例
    :param srs_whip_url: SRS服务器的W-HIP URL
    """
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    import aiohttp
    async with aiohttp.ClientSession() as session:
        # 使用给定的WHIP地址
        async with session.post(srs_whip_url, data=pc.localDescription.sdp) as response:
            answer_data = await response.text()
            print(answer_data, "*****")
            answer = RTCSessionDescription(sdp=answer_data, type='answer')
            await pc.setRemoteDescription(answer)


async def main():
    srs_whip_url = "http://192.168.21.13:1985/rtc/v1/whip/?app=live&stream=livestream1&eip=192.168.21.13"
    pc = RTCPeerConnection()
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    video_track = VideoStreamTrack1()
    audio_track = AudioStreamTrack1()
    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    await create_offer_and_send(pc, srs_whip_url)

    # 保持程序运行，持续推送视频流，这里简单使用一个循环，可以根据实际需求改进退出机制等
    # while True:
    #     await asyncio.sleep(10)


if __name__ == "__main__":
    # asyncio.run(main())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()  