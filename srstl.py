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
import copy

r = redis.Redis(host='10.23.32.63', port=6389, password=None)

# 初始化一个空列表用于存储处理后的图片
temp_frames_arr = []

# 循环处理 500 张图片
for i in range(2):
    try:
        # 图片文件名
        #image_path = f"data/video/jingyin/{i}.png"
        image_path = "0.png"
        # 读取图片
        image = cv2.imread(image_path)
        if image is not None:
            # 调整图片大小
            resized_image = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
            # 将处理后的图片添加到列表中
            temp_frames_arr.append(resized_image)
        else:
            print(f"无法读取图片: {image_path}")
    except Exception as e:
        print(f"处理图片 {i}.png 时出错: {e}")

# 将列表转换为 numpy 数组
temp_frames_arr = np.array(temp_frames_arr)

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
        #image = self.green_screen_keying2(image)
        self.temp_frames = np.array(image)
        self.temp_frames2 = self.temp_frames
        self.temp_frames_arr = temp_frames_arr
        self.arr_flag = 1
        self.bofang = False
        self.audio_track = audio_track
      
        self.background = cv2.imread('./bg.png')
        self.background = cv2.resize(self.background, (int(self.background.shape[1] / 1.5), int(self.background.shape[0] / 1.5)))
        image = self.green_screen_keying2(image)

    def green_screen_keying2(self, image):
        background = copy.deepcopy(self.background)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # 反转掩码
        mask_inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(image, image, mask=mask_inv)
        height,width, channels = fg.shape

        start_x = 450
        start_y = 40


        #print(start_y, start_y + height, background.shape, start_x, start_x + width, "eeeeeee")

        background_region = background[start_y:start_y + height, start_x:start_x + width]
        #background_region = background[start_x:start_x + width, start_y:start_y + height]
        background_masked = cv2.bitwise_and(background_region, background_region, mask=mask)

        combined_region = cv2.bitwise_or(fg, background_masked)

        # 将合并后的区域放回背景图片中
        background[start_y:start_y + height, start_x:start_x + width] = combined_region

        # 保存融合后的图片
        #cv2.imwrite("./test_pg1.png", background)
        
        #return combined_region
        #print(self.background.shape[1], "**********")
        return background[0:715, start_x:start_x + width]
        #return background

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
            frame = self.temp_frames_arr[self.arr_flag]

            if self.audio_name is None:
                frame = self.temp_frames_arr[self.arr_flag]
                #再次获取
                temp_audio_name = r.get(self.zbjname)
                if temp_audio_name:
                    temp_audio_name = temp_audio_name.decode('utf-8')
                if temp_audio_name and temp_audio_name != self.audio_name:
                    self.audio_name = temp_audio_name #再次获取
                    self.video_num = int(r.get(self.audio_name))
                else:
                    self.video_num = 0
            else:
                if self.img_index >= self.video_num - 1:
                    if self.video_num != 0:
                        r.psetex(self.zbjname + "check", 360000, 1) # 不能超过6分钟没反应。
                    self.img_index = 0
                    self.bofang = False
                    frame = self.temp_frames_arr[self.arr_flag]

                    #再次获取
                    temp_audio_name = r.get(self.zbjname)
                    if temp_audio_name:
                        temp_audio_name = temp_audio_name.decode('utf-8')
                    if temp_audio_name and temp_audio_name != self.audio_name:
                        self.audio_name = temp_audio_name #再次获取
                        self.video_num = int(r.get(self.audio_name))
                    else:
                        self.video_num = 0
                else:
                    # key = self.audio_name + str(self.img_index)
                    # while r.exists(key) is False:  #直到key存在
                    #     time.sleep(1)
                    
                    # 不会掉帧
                    if self.audio_track.bofang:
                        self.bofang = True
                    if self.bofang:
                        key = self.audio_name + str(int(self.img_index))
                        val = r.get(key)
                        if val:
                            print(key, "********")
                            r.delete(key)
                            frame = pickle.loads(val)
                            self.temp_frames2 = frame
                        else:
                            frame = self.temp_frames2

                        self.img_index += 1

                    # 会掉帧
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
            self.arr_flag = (self.arr_flag + 1) % 500
            self.arr_flag = 1
            

            frame = self.green_screen_keying2(frame)
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))
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

        self.stream = None
        self.total_frames = 0
        self.video_num = 0
        self.audio_name = r.get(zbjname)
        if self.audio_name:
            self.audio_name = self.audio_name.decode('utf-8')

        self.current_frame = 0
        self.bofang = False
        
        
    async def recv(self):
        try:

            flag_data = False
            row_data= None
            if self.audio_name is None:

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
                # zbjname
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0
                    self.bofang = False

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
            
            frame = AudioFrame(format="s16", layout="mono", samples=self.frame_size)

            if flag_data and self.bofang:
                # 假设row_data是音频数据数组
                max_val = np.max(np.abs(row_data))
                if max_val > 1.0:
                    row_data = row_data / max_val
                # 现在row_data的范围在-1.0到1.0之间，可以进行量化
                row_data = (row_data * 32767).astype(np.int16)
                frame.planes[0].update(row_data.tobytes())
                
            else:
                # 按照特定格式处理为静音数据
                silent_bytes = bytes([0x00] * (self.frame_size * 2))  # s16类型每个采样占2字节
                frame.planes[0].update(silent_bytes)
                
                if self.audio_name and r.exists(self.audio_name + str(6)):
                    #r.set(self.zbjname + "ok", 1)
                    self.bofang = True

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
        srs_whip_url = f"http://{self.zbjip}/rtc/v1/whip/?app=live&stream={self.zbjname}&eip={self.zbjip}&secret=4adbc3be84cf4d39851cf2dd1f91f827"
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
        while True:
            if self.zbjname != 'hnkjxyms' and (self.stop or not r.exists(self.zbjname + 'check')):
                print(self.zbjname + "直播间关闭了")
                await pc.close()
                break
            await asyncio.sleep(1)

    def run(self):
        asyncio.run(self.main())
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # loop.run_until_complete(self.main())
        # loop.run_forever() 

# if __name__ == "__main__":
#     # asyncio.run(main())
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(main())
#     loop.run_forever()  
