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
# import pickle

r = redis.Redis(host='localhost', port=6379, password=None)

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

# def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
#     cap = cv2.VideoCapture(vid_path)
#     count = 0
#     while True:
#         if count > cut_frame:
#             break
#         ret, frame = cap.read()
#         if ret:
#             cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
#             count += 1
#         else:
#             break

# def osmakedirs(path_list):
#     for path in path_list:
#         os.makedirs(path) if not os.path.exists(path) else None
    

@torch.no_grad() 
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
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
    
    
    # def process_frames(self, 
    #                    res_frame_queue,
    #                    video_len,
    #                    skip_save_images):
    #     print(video_len)
    #     while True:
    #         if self.idx>=video_len-1:
    #             break
    #         try:
    #             start = time.time()
    #             res_frame = res_frame_queue.get(block=True, timeout=1)
    #         except queue.Empty:
    #             continue
      
    #         bbox = self.coord_list_cycle[self.idx%(len(self.coord_list_cycle))]
    #         ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx%(len(self.frame_list_cycle))])
    #         x1, y1, x2, y2 = bbox
    #         try:
    #             res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    #         except:
    #             continue
    #         mask = self.mask_list_cycle[self.idx%(len(self.mask_list_cycle))]
    #         mask_crop_box = self.mask_coords_list_cycle[self.idx%(len(self.mask_coords_list_cycle))]
    #         #combine_frame = get_image(ori_frame,res_frame,bbox)
    #         combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

    #         if skip_save_images is False:
    #             cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png",combine_frame)
    #         self.idx = self.idx + 1
    
    def process_frames2(self, 
                       res_frame_queue,
                       flag,
                       video_len):
        print(video_len)
        flag_batch = flag + self.batch_size
        while True:
            print(flag, ")))))))")
            if flag>=video_len-1 or flag >= flag_batch:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
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
            #combine_frame = get_image(ori_frame,res_frame,bbox)
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            # if skip_save_images is False:
            cv2.imwrite(f"{self.avatar_path}/tmp/{str(flag)}.png",combine_frame)
            flag = flag + 1

    # def inference(self, 
    #               audio_path, 
    #               out_vid_name, 
    #               fps,
    #               skip_save_images):
    #     os.makedirs(self.avatar_path+'/tmp',exist_ok =True)   
    #     print("start inference")
    #     ############################################## extract audio feature ##############################################
    #     start_time = time.time()
    #     whisper_feature = audio_processor.audio2feat(audio_path)
    #     whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    #     print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
    #     ############################################## inference batch by batch ##############################################
    #     video_num = len(whisper_chunks)   
    #     res_frame_queue = queue.Queue()
    #     self.idx = 0
    #     # # Create a sub-thread and start it
    #     process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
    #     process_thread.start()

    #     gen = datagen(whisper_chunks,
    #                   self.input_latent_list_cycle, 
    #                   self.batch_size)
    #     start_time = time.time()
        
    #     for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/self.batch_size)))):
    #         print(len(whisper_batch), len(latent_batch), "!!!!!!!!!")
    #         print(whisper_batch, "PPPPPPPPP")
    #         print(latent_batch, "LLLLLLLLLL")
    #         audio_feature_batch = torch.from_numpy(whisper_batch)
    #         audio_feature_batch = audio_feature_batch.to(device=unet.device,
    #                                                      dtype=unet.model.dtype)
    #         audio_feature_batch = pe(audio_feature_batch)
    #         latent_batch = latent_batch.to(dtype=unet.model.dtype)

    #         pred_latents = unet.model(latent_batch, 
    #                                   timesteps, 
    #                                   encoder_hidden_states=audio_feature_batch).sample
    #         recon = vae.decode_latents(pred_latents)
    #         for res_frame in recon:
    #             res_frame_queue.put(res_frame)
    #         break
    #     # Close the queue and sub-thread after all tasks are completed
    #     process_thread.join()
    
    def infer(self, user_id):
        video_num = int(r.get(user_id + '_all'))
        while True:
            whisper_batch, latent_batch = [], []

            flag = r.incr(user_id + 'counter', self.batch_size) - self.batch_size
            print(flag, self.batch_size)
            for i in range(self.batch_size):
                key_str = user_id + '_' + str(flag + i)
                if flag + i >= video_num:
                    r.set(user_id + 'counter', 1000000)
                    break
                temp = r.get(key_str)
                if temp is not None:
                    print(key_str)
                    # r.delete(key_str)
                    whisper_batch.append(pickle.loads(temp))
                    idx = (i+flag)%len(self.input_latent_list_cycle)
                    latent = self.input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                else:
                    r.set(user_id + 'counter', 1000000)
                    break
            if len(whisper_batch) == 0:
                time.sleep(10)
                r.set(user_id + 'counter', 1000000)
                break
            print(len(whisper_batch), "??????")
            res_frame_queue = queue.Queue()
            process_thread = threading.Thread(target=self.process_frames2, args=(res_frame_queue, flag, video_num))
            process_thread.start()

            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            # print(whisper_batch, "###########")
            # print(latent_batch, "$$$$$$$$$$$$$$$")

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
        avatar_id = 'avator_6', 
        video_path = 'data/video/output_video.mp4', 
        bbox_shift = -7, 
        batch_size = args.batch_size,
        preparation= False)
    
    user_id = 'sang'
    audio_url = "output_audio.wav"
    
    whisper_feature = audio_processor.audio2feat(audio_url)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=args.fps)

    print(len(whisper_chunks))

    whisper_batch = []
    r.set(user_id + '_all', len(whisper_chunks))
    for i, w in enumerate(whisper_chunks):
        # if i == 0:
        #     print(w)
        serialized_w = pickle.dumps(w)

        r.set(user_id + 'counter', 0)
        r.set(user_id + '_' + str(i), serialized_w)
    
    avatar.infer(user_id)

    # avatar.inference('data/audio/sun.wav', 
    #                         1, 
    #                         args.fps,
    #                         args.skip_save_images)