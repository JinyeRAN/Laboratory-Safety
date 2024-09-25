import cv2
import torch
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


def video2frame(videos_path, frames_save_path, time_interval):
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            if isinstance(image, np.ndarray):
                cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)


def Composite_video(im_list, video_dir, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 

    conta = list(map(lambda x:os.path.join(im_list, 'frame'+str(x+1)+'.jpg'), [i for i in range(len(os.listdir(im_list)))]))

    img = Image.open(conta[0])
    img_size = img.size

    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for il in conta:
        im_name = il
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
    videoWriter.release()


def detect_process(frame_dir, frame_dir_output, model):
    for filename in os.listdir(frame_dir):
        fileroot = os.path.join(frame_dir, filename)
        img = Image.open(fileroot).convert('RGB')

        g = (640 / max(img.size))
        img = img.resize((int(x * g) for x in img.size), Image.ANTIALIAS)
        results = model(img)
        results.render()
        img = Image.fromarray(results.ims[0])

        fileroot_output = os.path.join(frame_dir_output, filename)
        img.save(fileroot_output)


if __name__=='__main__':
    import os
    model = torch.hub.load(repo_or_dir='./ultralytics-yolov5', source='local', model='custom', path='best.pt')

    video_dir = './data/fire/demo.mp4'
    frame_dir_ouput = './data/fire_result/'
    frame_dir = './data/fire_frame'
    video_dir_ouput = './data/fire/demo.mp4'
    
    video2frame(video_dir, frame_dir, time_interval=1)
    detect_process(frame_dir, frame_dir_output, model)
    Composite_video(frame_dir_output, video_dir_ouput)