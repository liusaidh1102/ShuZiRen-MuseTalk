import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model
import shutil

import threading
import queue

import time
import redis

r = redis.Redis(host='10.23.32.63', port=6389, password=None)

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

@torch.no_grad() 
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size,queue_str, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id":avatar_id,
            "video_path":video_path,
            "bbox_shift":bbox_shift   
        }
        self.preparation = False
        self.batch_size = batch_size
        self.queue_str = queue_str
        self.idx = 0
        self.init()
        
    def init(self):
        if not os.path.exists(self.avatar_path):
            print(f"{self.avatar_id} does not exist, you should set preparation to True")
            sys.exit()

        with open(self.avatar_info_path, "r") as f:
            avatar_info = json.load(f)
            

        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
    def green_screen_keying(self, image, background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (int(background.shape[1] / 1.5), int(background.shape[0] / 1.5)))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # 反转掩码
        mask_inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(image, image, mask=mask_inv)
        height, width, channels = fg.shape

        start_x = 450
        start_y = 40

        background_region = background[start_y:start_y + height, start_x:start_x + width]

        background_masked = cv2.bitwise_and(background_region, background_region, mask=mask)

        combined_region = cv2.bitwise_or(fg, background_masked)

        # 将合并后的区域放回背景图片中
        background[start_y:start_y + height, start_x:start_x + width] = combined_region

        # 保存融合后的图片
        # cv2.imwrite(result_path, background)

        return background
    
    def process_frames2(self, 
                       res_frame_queue,
                       flag,
                       video_len,audio_name):
        print(video_len)
        flag_batch = flag + self.batch_size
        try_count = 1
        while True:
            print(flag, ")))))))")
            if flag>=video_len-1 or flag >= flag_batch or try_count > 5:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                try_count += 1
                continue
      
            bbox = self.coord_list_cycle[flag%(len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[flag%(len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            mask = self.mask_list_cycle[flag%(len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[flag%(len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
            
            combine_frame = cv2.resize(combine_frame,(int(combine_frame.shape[1] / 3), int(combine_frame.shape[0] / 3)))
            #combine_frame = self.green_screen_keying(combine_frame, "./bg.jpg")
            # 是否需要保存图片
            # cv2.imwrite(f"{self.avatar_path}/tmp/{str(flag)}.png",combine_frame)
            # r.set(audio_name + str(flag), pickle.dumps(combine_frame))
            r.psetex(audio_name + str(flag), 300000, pickle.dumps(combine_frame))
            flag = flag + 1
    
    def infer(self):
        # video_num = int(r.get(user_id + '_all'))
        while True:
            print(123)
            queue_result = r.blpop(self.queue_str, timeout=0)
            if queue_result:
                # 因为blpop返回的是一个包含键名和值的元组，所以取第二个元素为实际数据
                element = str(queue_result[1].decode('utf-8')).split("_")
                print(f"从队列中取出元素: {element}")
                audio_name = element[2]
                video_num = int(element[0])
                flag = int(element[1])
            else:
                print("队列为空，继续等待...")
                continue

            whisper_batch, latent_batch = [], []

            # flag = r.incr(user_id + 'counter', self.batch_size) - self.batch_size
            print(flag, self.batch_size)
            for i in range(self.batch_size):
                key_str = audio_name + '_' + str(flag + i)
                if flag + i >= video_num:
                    # r.set(user_id + 'counter', 1000000)
                    break
                temp = r.get(key_str)
                if temp is not None:
                    print(key_str)
                    r.delete(key_str)
                    whisper_batch.append(pickle.loads(temp))
                    idx = (i+flag)%len(self.input_latent_list_cycle)
                    latent = self.input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                else:
                    # r.set(user_id + 'counter', 1000000)
                    break
            if len(whisper_batch) == 0:
                # time.sleep(10)
                # r.set(user_id + 'counter', 1000000)
                continue
            print(len(whisper_batch), "??????")
            res_frame_queue = queue.Queue()
            process_thread = threading.Thread(target=self.process_frames2, args=(res_frame_queue, flag, video_num,audio_name))
            process_thread.start()

            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)

            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch, 
                                      timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)

            process_thread.join()
    

if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", 
                        type=str, 
                        default="configs/inference/realtime.yaml",
    )
    parser.add_argument("--queue",
                        type=str,
                        default="queue1",
    )
    parser.add_argument("--fps", 
                        type=int, 
                        default=25,
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4,
    )
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether skip saving images for better generation speed calculation",
    )

    args = parser.parse_args()
    

    avatar = Avatar(
        avatar_id = 'avator_1', 
        video_path = 'data/video/output_video.mp4', 
        bbox_shift = -7, 
        batch_size = args.batch_size,
        queue_str = args.queue,
        preparation= False)
    
    # user_id = 'sang'
    # audio_url = "output_audio.wav"
    
    # whisper_feature = audio_processor.audio2feat(audio_url)
    # whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=args.fps)

    # print(len(whisper_chunks))

    # whisper_batch = []
    # r.set(user_id + '_all', len(whisper_chunks))
    # for i, w in enumerate(whisper_chunks):
    #     # if i == 0:
    #     #     print(w)
    #     serialized_w = pickle.dumps(w)

    #     r.set(user_id + 'counter', 0)
    #     r.set(user_id + '_' + str(i), serialized_w)
    
    avatar.infer()

    # avatar.inference('data/audio/sun.wav', 
    #                         1, 
    #                         args.fps,
    #                         args.skip_save_images)
