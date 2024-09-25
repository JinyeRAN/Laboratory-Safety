import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tools import myThread, CurrentTime, list_labid
from detect_sub import YoLov5TRTN
from detect_main import YoLov5TRT
from tracker_alarm import Alarm
from copy import deepcopy

import cv2
import json
from queue import Queue
import numpy as np
import time
from threading import Thread
import subprocess
from flask_cors import *
from flask import Flask, request, current_app

from deep_sort_pytorch.deep_sort import DeepSort
from Exp_detect_v1 import Exp

# app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False
# CORS(app, resources=r'/*')
# app.debug = False
# #
# # @app.route('/video/control', methods=['post'])
# # def post_http():
# #     labID,  state = request.form.get('state')
# #     rtmp = cameraID_RTMP[labID]
# #     model = exec('Stream' + labID)
# #     if(state == 'start'):
# #         model.push_stream(rtmp, INPUT_W=None, INPUT_H=None)
# #     elif (state == 'stop'):
# #         model.stop_stream()
# #     return state
# @app.route('/video/init', methods=['post'])
# @cross_origin()
# def post_http():
#     data = request.form.get('data')
#     data = json.loads(data)
#     print(data, flush=True)

#     if not 'OPS' in data.keys():
#         rtmp = data['EQUIPMENT_ID']
#         rtmp_f = "rtmp://10.3.50.250:1935/boteng/" + str(rtmp)
#         work_time = 0
#         algorithm = data['EQUIPMENT_ALGORITHM']
#         if algorithm == 'PPE检测_':
#             algorithm = 'ppe'
#         elif algorithm =='实验现象_':
#             algorithm = 'diff'
#         rtsp = data['EQUIPMENT_IP']
#         rtsp = "rtsp://admin:hik12345@" + rtsp + "/Streaming/Channels/1"
#         labid = data['EQUIPMENT_POSITION']
#         lab_list.add_elements(rtmp)
#         streamname = lab_list.utlize1(rtmp)
#         exec(streamname + ' = MainThread(cfg, labid, rtsp, rtmp, rtmp_f, work_time,algorithm)')

#         lab_list.add_func(rtmp, eval(streamname))

#         # eval('global streamname')
#         eval('myThread(' + streamname + '.capture, []).start()')
#         if algorithm=='ppe':
#             eval('myThread(' + streamname + '.' + algorithm + ',[detect_Q,]).start()')
#         elif algorithm=='diff':
#             print('diff ',streamname, flush=True)
#             eval('myThread(' + streamname + '.' + algorithm + ',[]).start()')
#         # eval('myThread(' + streamname + '.push_stream, []).start()')
#         #rtmp = data['labid']
#         streamname = lab_list.utlize(rtmp)
#         myThread(streamname.push_stream, []).start()
#     else:
#         ops = data['OPS']
#         if ops == 'del':
#             rtmp = data['labid']
#             streamname = lab_list.utlize(rtmp)
#             streamname.stop_thread()
#             del(streamname)
#             lab_list.delete_elements(rtmp)
#             lab_list.del_func(rtmp)

#         elif ops == 'push':
#             pass
#             # rtmp = data['labid']
#             # streamname = lab_list.utlize(rtmp)
#             # myThread(streamname.push_stream, []).start()

#         elif ops == 'stop':
#             rtmp = data['labid']
#             streamname = lab_list.utlize(rtmp)
#             streamname.stop_stream()
#         else:
#             pass

#     return 'succuss'


class MainThread(object):
    def __init__(self, cfg, labID, rtsp, rtmp, rtmpf, work_time,algorithm):
        self.idd = rtmp
        streamname = lab_list.utlize1(rtmp)
        self.VideoFrameQueue = Queue()
        self.PushFrameQueue = Queue()

        self.rtsp = rtsp
        self.rtmp = rtmpf

        self.DecetectionPeriods = 15
        self.DecetectionPeriods_F = 15
        self.Point = 100
        self.labID = labID
        self.Sign = True

        # set time
        self.work_time = work_time
        if work_time == True:
            self.begintime1 = 83000
            self.aftertime1 = 120000
            self.begintime2 = 130000
            self.aftertime2 = 170000
        else:
            self.aftertime = None
            self.begintime = None
        if algorithm=='ppe':
            self.alarm = Alarm()
            self.tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                     max_dist=cfg.DEEPSORT.MAX_DIST,
                                     min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                     max_age=cfg.DEEPSORT.MAX_AGE,
                                     n_init=cfg.DEEPSORT.N_INIT,
                                     nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                     use_cuda=True)
        elif algorithm=='diff':
            self.camera = Exp(str(rtmp))
        self.streamname = streamname
        self.stop = False
        # diff


    def capture(self):
        # FPS = 22
        cap = cv2.VideoCapture(self.rtsp)
        while not self.stop:
            if cap.isOpened():
                _, frame = cap.read()
                self.VideoFrameQueue.put(frame)
                if self.VideoFrameQueue.qsize() > 2:
                    self.VideoFrameQueue.get()
                else:
                    pass
            else:
                cap = cv2.VideoCapture(self.rtsp)

    def ppe(self, detect_Q):
        while not self.stop:
            Img = self.VideoFrameQueue.get()
            img1 = deepcopy(Img)
            if Img is None:
                print('acquire image raise error')
                pass
            else:
                # pass
                ImgPushStream, result_boxes, result_scores, result_classid = TRT_People.infer(Img, self.streamname)

                self.PushFrameQueue.put(ImgPushStream)
                if self.PushFrameQueue.qsize() > 2:
                    self.PushFrameQueue.get()

                if self.work_time:
                    if self.begintime1 < CurrentTime() < self.aftertime1 or self.begintime2 < CurrentTime() < self.aftertime2:

                        if not self.Point == 0:
                            self.Point -= 1
                        else:
                            self.Point = 100
                            track = self.tracker.update(result_boxes, result_scores, result_classid, img1)
                            if self.DecetectionPeriods > 0:
                                self.DecetectionPeriods -= 1
                            else:
                                self.DecetectionPeriods = self.DecetectionPeriods_F
                                if not len(track) < 1:
                                    detect_Q.put([self.streamname, track, img1])
                else:
                    if not self.Point == 0:
                        self.Point -= 1
                    else:
                        self.Point = 10
                        track = self.tracker.update(result_boxes, result_scores, result_classid, Img)
                        if self.DecetectionPeriods > 0:
                            self.DecetectionPeriods -= 1
                        else:
                            self.DecetectionPeriods = self.DecetectionPeriods_F
                            detect_Q.put([self.idd, track, Img])
                            if not len(track)<1:
                                detect_Q.put([self.streamname, track, img1])

    def diff(self):
        while not self.stop:
            frame = self.VideoFrameQueue.get()
            image_with_mask = self.camera.get_vessel_image_with_mask(frame)
            self.PushFrameQueue.put(image_with_mask)

    def detect_sub(self, track, Img):
        WhiteCoat = TRT_Labcoat.infer(Img)
        glass = TRT_Glasses.infer(Img)
        self.alarm.alarm(track, [WhiteCoat, glass], Img, self.labID, self.rtmp)

    def push_stream(self, INPUT_W=640, INPUT_H=320):
        self.Sign = True
        command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', str(INPUT_W) + 'x' + str(INPUT_H), '-r', '21', '-i', '-', '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-f', 'flv', '-flvflags', 'no_duration_filesize',
                   self.rtmp]
        pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
        while self.Sign:
            img = self.PushFrameQueue.get()
            img = cv2.resize(img, (INPUT_W, INPUT_H))
            pipe.stdin.write(img.tobytes())
        pipe.terminate()

    def stop_stream(self):
        self.Sign = False

    def stop_thread(self):
        self.stop = True


def detect_process(detect_Q):
    while True:
        if detect_Q.qsize() == 0:
            time.sleep(1)
            pass
        elif detect_Q.qsize() == 1:
            while detect_Q.qsize() == 1:
                data = detect_Q.get()
                streamname, track, img = data[0], data[1], data[2]
                streamname = lab_list.utlize(streamname)
            streamname.detect_sub(track, img)
        elif detect_Q.qsize() > 1:
            print('process')
            data = detect_Q.get()
            streamname, track, img = data[0], data[1], data[2]
            streamname = lab_list.utlize(streamname)
            streamname.detect_sub(track, img)


if __name__ == '__main__':
    import argparse
    import os
    import sys
    from deep_sort_pytorch.utils.parser import get_config

    path = os.path.split(os.path.realpath(sys.argv[0]))[0] # 绝对路径
    # acquire tracker init
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_deepsort", type=str, default=path + "/deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    # fenye
    inteval_time_detect_vessl = 600  # Unit: Second
    count_for_detect_vessel = 0

    # init trt model
    TRT_People = YoLov5TRT(path + "/yolov5trt/build/libmyplugins.so",path + "/yolov5trt/people-v4.engine")
    # TRT_Labcoat = YoLov5TRTN(path + "/yolov5trt/build/libmyplugins.so",path + "/yolov5trt/labcoat-v4.engine")
    # TRT_Glasses = YoLov5TRTN(path + "/yolov5trt/build/libmyplugins.so",path + "/yolov5trt/glasses-v4.engine")
    # TRT_Gloves = YoLov5TRTN(path + "/yolov5trt/build/libmyplugins.so",path + "/yolov5trt/gloves-v4.engine")

    detect_Q = Queue()
    myThread(detect_process, [detect_Q]).start()
    # # init table
    cameraID_RTSP = {
        'lab1': "rtsp://admin:hik12345@10.3.50.21/Streaming/Channels/1",
    }

    cameraID_RTMP = {
        'lab1': "rtmp://10.3.50.250:1935/boteng/123",
    }

    # 'ppe'
    ID = 'lab1'
    # add pro
    lab_list = list_labid()
    rtmp = '123'
    lab_list.add_elements(rtmp)
    rtsp_1 = "rtsp://admin:traffic0031!@192.168.1.102/Streaming/Channels/1"
    rtmpf_1 = "rtmp://192.168.3.231:1935/myapp/123"
    # rtsp_1 = test_video
    Stream1 = MainThread(cfg, labID=ID, rtsp=rtsp_1, rtmp=rtmp, rtmpf=rtmpf_1, work_time=True, algorithm='ppe')

    myThread(Stream1.capture, []).start()
    myThread(Stream1.ppe, [detect_Q, ]).start()
    myThread(Stream1.push_stream, []).start()

    # lab_list = list_labid()
    # labID2Stream = {
    #     'lab1': "Stream1",
    # }
    # app.run(host='172.17.0.2', port=1234, threaded=True) #

    TRT_People.destroy()
    # TRT_Labcoat.destroy()
    # TRT_Glasses.destroy()
    # TRT_Glasses.destroy()

