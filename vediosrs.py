import asyncio
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


class VideoPusher:
    def __init__(self, zbjip, zbjname):
        self.zbjip = zbjip
        self.zbjname = zbjname

    async def push_video(self, video_file_path):
        srs_whip_url = f"http://{self.zbjip}/rtc/v1/whip/?app=live&stream={self.zbjname}&eip={self.zbjip}&secret=4adbc3be84cf4d39851cf2dd1f91f827"
        print(srs_whip_url)
        # 创建 RTCPeerConnection 对象
        pc = RTCPeerConnection()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(srs_whip_url)
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()

        while True:
            # 创建 MediaPlayer 加载视频文件
            player = MediaPlayer(video_file_path)
            # 添加视频轨道到连接
            sender = None
            if player.video:
                sender = pc.addTrack(player.video)

            # 创建 SDP 描述
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # 使用 aiohttp 发送 POST 请求，将本地 SDP 发送到 SRS 的 WHIP URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    srs_whip_url,
                    headers={"Content-Type": "application/sdp"},
                    data=pc.localDescription.sdp
                ) as response:
                    if response.status == 201:
                        answer_sdp = await response.text()
                        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                        await pc.setRemoteDescription(answer)
                    else:
                        print(f"Failed to push stream. Status code: {response.status}")
                        return

            print("kaishibofang")
            # 等待视频播放结束的事件
            ended_event = asyncio.Event()

            def on_track_ended():
                ended_event.set()

            if player.video:
                player.video.on("ended", on_track_ended)

            try:
                await ended_event.wait()
            except asyncio.CancelledError:
                pass
            finally:
                # 停止播放视频
                if hasattr(player, '_stop') and player.video:
                    player._stop(player.video)
                # 停止发送器
                if sender:
                    await sender.stop()


async def main():
    pusher = VideoPusher(zbjip="srs.xiaozhu.com:2022", zbjname="test")
    await pusher.push_video("tests/1004-r.mp4")


if __name__ == "__main__":
    asyncio.run(main())
    
