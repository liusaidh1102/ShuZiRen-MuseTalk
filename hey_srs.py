import asyncio
#import logging
import requests
import os

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import redis
r = redis.Redis(host='10.23.32.63', port=6389, password=None)
class HumanSRSv2:
    def __init__(self, zbjname, zbjip):
        self.zbjname = zbjname
        self.stop = False
        self.zbjip = zbjip

    async def send_video(self, player):
        # player = MediaPlayer(file_path)
        pc = RTCPeerConnection()
        # pc.addTrack(player.video)
        # pc.addTrack(player.audio)
        video_track = player.video
        audio_track = player.audio
        # 确保音频和视频轨道准备好
        while not (video_track.readyState == "live" and audio_track.readyState == "live"):
            await asyncio.sleep(0.1)

        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # SRS服务器地址192.168.1.100
        # livestream000001可以替换为任意字符串
        srs_whip_url = f"http://{self.zbjip}/rtc/v1/whip/?app=live&stream={self.zbjname}&eip={self.zbjip}&secret=4adbc3be84cf4d39851cf2dd1f91f827"
        resp = requests.post(
            srs_whip_url,
            headers={"Content-type": "application/sdp"},
            data=offer.sdp,
        )
        await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))

        while True and r.exists(self.zbjname + 'check'):
            await asyncio.sleep(0.2)
            print(f"{player.video.readyState}")
            if player.video.readyState == "ended":
                break

        await pc.close()

    async def main(self):
        while True and r.exists(self.zbjname + 'check'):
            value = r.get(self.zbjname + "_hey_flag")
            if value is not None:
                # 将字节类型转换为字符串类型（如果值是字符串）
                value = value.decode('utf-8')
                print(f"取出的值为: {value}")
                # 删除键
                if value not in ['98','99','97']:
                    filename = "tests/" + self.zbjname + "_" + value + ".mp4"
                else :
                    filename = "tests/" + value + ".mp4"
                print(filename)
                if os.path.exists(filename):
                    print(f"文件 {filename} 存在。")
                    await self.send_video(MediaPlayer(filename))
                else:
                    print(f"文件 {filename} 不存在。")
                    await self.send_video(MediaPlayer("tests/laoshi1-6.mp4"))
                r.delete(self.zbjname + "_hey_flag")
                print(f"键已成功删除")
            else:
                print(f"不存在")
                await self.send_video(MediaPlayer("tests/laoshi1-6.mp4"))

    def run(self):
        #logging.basicConfig(level=logging.ERROR)
        asyncio.run(self.main())
