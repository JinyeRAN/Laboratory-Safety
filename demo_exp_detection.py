#........................Imports..............................
# from tkinter.constants import N
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.animation as animation
import time

# from numpy.core.fromnumeric import argmax
from Vessel_detect.Vessel_detect import get_frame_OutAnnMap
# import sys
import pandas as pd
import os

class Exp():
    def __init__(self,webcamer_id:str,
                interval_time_detect_vessel = 600, # Unit: Second
                interval_time_calculate_image_entropy = 1, # Unit: Second
                #interval_time_detect_vessel_while_no_vessel_detect = 10, #Unit: Second 
                video_stream_fps = 25,
                default_save_data_format = 'xlsx', # optional format:csv
                count_for_detect_vessel = 0,
                count_for_calculate_entropy= 0,
                #count_for_detect_vessel_while_no_vessel_detect = 0
                ):

        self.id = webcamer_id
        self.inteval_time_detect_vessel = interval_time_detect_vessel * video_stream_fps
        self.interval_time_calculate_image_entropy = interval_time_calculate_image_entropy * video_stream_fps
        #self.interval_time_detect_vessel_while_no_vessel_detect = interval_time_detect_vessel_while_no_vessel_detect*video_stream_fps
        self.video_stream_fps = video_stream_fps
        self.default_save_data_format = default_save_data_format
        self.count_for_detect_vessel = count_for_detect_vessel
        self.count_for_calculate_entropy = count_for_calculate_entropy
        #self.count_for_detect_vessel_while_no_vessel_detect = count_for_detect_vessel_while_no_vessel_detect
        

        self.output_dir = 'output/'+ webcamer_id +'/'
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.liquid_sep_video_clip = []
        self.liquid_sep_entropy_clip = []        

    def liquid_separation_detect(self,img,mask):
        "Create output dir for liquid_separation_detect"

        self.liquid_sep_output_dir = self.output_dir + 'liquid_separation_detect/'
        if not os.path.exists(self.liquid_sep_output_dir): os.makedirs(self.liquid_sep_output_dir)

        # Get frame/entropy data, and put them into a video/entropy clip.
        entropy = self.cal_1D_entropy(img,mask)
        if len(self.liquid_sep_entropy_clip) < 40:
            self.liquid_sep_video_clip.append(img)
            self.liquid_sep_entropy_clip.append(entropy)
        else:
            clip = self.liquid_sep_entropy_clip[15:20]
            m = max(self.liquid_sep_entropy_clip) - min(self.liquid_sep_entropy_clip)
            if all(clip[i]<clip[i+1] for i in range(len(clip)-1)) and m > 0.02:
                print("Detect Liquid Separation Process! Save Data in the output dir!")
                self.save_liquid_separation_results(self.liquid_sep_video_clip,self.liquid_sep_entropy_clip)
                self.liquid_sep_entropy_clip = []
                self.liquid_sep_video_clip = []

            else:
                self.liquid_sep_video_clip.append(img)
                self.liquid_sep_entropy_clip.append(entropy)
                del self.liquid_sep_video_clip[0]
                del self.liquid_sep_entropy_clip[0]

    def get_vessel_image_with_mask(self,frame):
        "Input vessel image and return image with mask(trigger liquid_separation_detect)"
        
        global value

        if self.count_for_detect_vessel % self.inteval_time_detect_vessel == 0:
            value,mask = get_frame_OutAnnMap(frame)
            self.mask = mask.copy() 

        h,w = np.shape(self.mask)
        resized_frame = cv2.resize(frame,(w,h),interpolation= cv2.INTER_AREA)
        self.resized_frame = resized_frame.copy() # save resized_frame as the input of entropy calculation
        
        if value is False:
                print("No vessel detect! Program starts to try again")
                self.count_for_detect_vessel = -1
                image_with_mask = frame
        else:
            img1 = self.mask.copy()
            img2 = self.resized_frame.copy()
            
            img2_bg = cv2.bitwise_and(img2,img2,mask = img1)
            img_composed = cv2.addWeighted(img2_bg,0.7,img2,0.3,0)
            ##################
            image_with_mask = img_composed
        
        if self.count_for_calculate_entropy % self.interval_time_calculate_image_entropy == 0:
            self.liquid_separation_detect(self.resized_frame,self.mask)


        self.count_for_detect_vessel += 1
        self.count_for_calculate_entropy += 1

        return image_with_mask
        
    def save_liquid_separation_results(self,video_clip,entropy_clip):
        """Save original frames into output dirs"""

        now_time = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())

        output_dir = self.liquid_sep_output_dir + now_time[:-9] + '/' + now_time[-8:] + '/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        #....................Save Entropy Clip For Liquid Separation Detect......

        if self.default_save_data_format == 'xlsx':
            entropy_clip_df = pd.DataFrame(entropy_clip) # entropy_clip shape is (data_num,2); entropy_clip[1][0]:time_serise; entropy_clip[1][1]:entropy
            
            writer = pd.ExcelWriter(output_dir + now_time + '.xlsx')
            entropy_clip_df.to_excel(writer,'page_1',float_format='%.5f')
            writer.save()
        elif self.default_save_data_format == 'csv':
            np.savetxt('data.csv',entropy_clip,delimiter = '')
        else:
            print("Save entropy_clip failed!\nPlease check data file format")

        #....................Save Video Clip For Liquid Separation Detect......

        frames_num,h,w,color_spaces_num = np.shape(video_clip) # video_clip shape [frames_num,height,width,color_spaces_num]
        out = cv2.VideoWriter(output_dir + now_time + '.mp4',\
            cv2.VideoWriter_fourcc(*'mp4v'),self.video_stream_fps,(w,h),True)

        for i in range(len(video_clip)):
            out.write(video_clip[i])
        out.release()

    def cal_1D_entropy(self,img,mask):
        hist_cv = cv2.calcHist([img],[0],mask,[256],[0,256])
        P = hist_cv/(len(img)*len(img[0])) 
        E = -np.sum([p *np.log2(p + 1e-5) for p in P])
        return E