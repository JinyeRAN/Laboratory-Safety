import os
import cv2
import sys
import torch
import argparse
import subprocess
import numpy as np
from queue import Queue
from threading import Thread

sys.path.insert(0, './yolov5')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# temp
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def judge(track_result,
          det):
    result = []
    count = 0
    for people in track_result:
        people_bbox = people[0:4]
        for bbox in det:
            catlogy_bbox = bbox[0:4]
            if compute_IOU(people_bbox,catlogy_bbox)>0:
                result.append(1)
                break
        if len(result) == count:
            count += 1
            result.append(0)
            continue
        count += 1
    return result


def compute_IOU(rec1,
                rec2):
    left_column_max = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max = max(rec1[1],rec2[1])
    down_row_min = min(rec1[3],rec2[3])
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


def video_capture(FrameOrlGetQueue,
                  FrameProcessGetQueue):
    # c = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # c = c+1
        if not ret:
            break
        # origin data
        FrameOrlGetQueue.put(frame)
        # process data
        FrameProcess = letterbox(frame, 640, stride=32, auto=True)[0]
        FrameProcess = FrameProcess.transpose((2, 0, 1))[::-1]
        FrameProcess = np.ascontiguousarray(FrameProcess)
        FrameProcessGetQueue.put(FrameProcess)
        # if c==6:
        #     break
    cap.release()


def Alarm(FrameProcessTrackerQueue,
          FrameOrlTrackerQueue,
          DetectionsTrackerQueue,
          FrameProcessAlarmQueue,
          FrameOrlAlarmQueue,
          DetectionsAlarmQueue,
          DetectionsQueue,
          AlarmQueue):
    Count = int(FrameNum)
    while cap.isOpened():
        # detection
        if Count != 0:
            if not FrameProcessInfQueue.empty() and not FrameOrlInfQueue.empty() and not DetectionsTrackerQueue.empty():
                FrameProcessAlarmQueue.put(FrameProcessTrackerQueue.get())
                FrameOrlAlarmQueue.put(FrameOrlTrackerQueue.get())
                DetectionsAlarmQueue.put(DetectionsTrackerQueue.get())
                Count = Count - 1
        else:
            # restore
            Count = int(FrameNum)
            # get data
            Image = FrameProcessInfQueue.get()
            Image0 = FrameOrlInfQueue.get()
            TrackerResult = DetectionsTrackerQueue.get()

            # preprocess
            Image = torch.from_numpy(Image).to(device)
            Image = Image.half() if half else Image.float()
            Image /= 255.0
            if Image.ndimension() == 3:
                Image = Image.unsqueeze(0)

            # Inference
            pred_people = model_people(Image, augment=args.augment)[0]
            pred_white_coat = model_white_coat(Image, augment=args.augment)[0]
            pred_glasses = model_glasses(Image, augment=args.augment)[0]
            pred_gloves = model_gloves(Image, augment=args.augment)[0]
            pred_face_mask = model_face_mask(Image, augment=args.augment)[0]

            # Apply NMS
            pred_people = non_max_suppression(
                pred_people, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            pred_white_coat = non_max_suppression(
                pred_white_coat, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            pred_glasses = non_max_suppression(
                pred_glasses, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            pred_gloves = non_max_suppression(
                pred_gloves, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            pred_face_mask = non_max_suppression(
                pred_face_mask, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            predict = [pred_people[0], pred_white_coat[0], pred_glasses[0], pred_gloves[0], pred_face_mask[0]]
            predict_process = []
            for item in predict:
                item[:, :4] = scale_coords(Image.shape[2:], item[:, :4], Image0.shape).round()
                predict_process.append(item)

            if len(predict) > 2 and len(TrackerResult) != 0:
                FinalResult = []
                for other in predict[1:]:
                    other = scale_coords(Image.shape[2:], other[:, :4], Image0.shape).round()
                    FinalResult.append(judge(TrackerResult, other))
                result = np.concatenate((TrackerResult, np.array(FinalResult).transpose()), axis=1)
                AlarmQueue.put(result)
            DetectionsQueue.put(predict_process)

            # restore queue
            FrameProcessAlarmQueue.put(Image)
            FrameOrlAlarmQueue.put(Image0)
            DetectionsAlarmQueue.put(TrackerResult)


def Tracker(FrameProcessGetQueue,
            FrameOrlGetQueue,
            FrameProcessInfQueue,
            FrameOrlInfQueue,
            DetectionsTrackerQueue):
    while cap.isOpened():
        # get data
        img = FrameProcessGetQueue.get()
        im0 = FrameOrlGetQueue.get()

        # preprocess
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred_people = model_people(img, augment=args.augment)[0]

        # Apply NMS
        predict = non_max_suppression(
            pred_people, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

        # track
        PeopleData = predict[0]
        PeopleData[:, :4] = scale_coords(img.shape[2:], PeopleData[:, :4], im0.shape).round()
        xywhs = xyxy2xywh(PeopleData[:, 0:4])
        confs = PeopleData[:, 4]
        clss = PeopleData[:, 5]

        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

        DetectionsTrackerQueue.put(outputs)
        FrameProcessInfQueue.put(img)
        FrameOrlInfQueue.put(im0)

    cap.release()


def AfterProcess(FrameOrlAlarmQueue,
          DetectionsAlarmQueue,
          DetectionsQueue,
          AlarmQueue):
    # while 1:
    while cap.isOpened():
        if not FrameOrlAlarmQueue.empty() and not DetectionsAlarmQueue.empty():
            im0, detection = FrameOrlAlarmQueue.get(), DetectionsAlarmQueue.get()
            if not AlarmQueue.empty():
                AlarmResult = AlarmQueue.get()

            if not DetectionsQueue.empty():
                DetectionsResult = DetectionsQueue.get()

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            C = 0
            for det in detection:
                for bbox in det:
                    bboxes = bbox[0:4]
                    conf = bbox[4]
                    label = f'{names[C]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(C, True))
                C = C + 1
            im0 = annotator.result()
            # plt.imshow(im0)
            # plt.show()
            # resize
            im0 = cv2.resize(im0, FrameOutSize)
            # write pipeline
            pipe.stdin.write(im0.tostring())
    cap.release()
    pipe.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--people_yolo_weights', nargs='+', type=str, default='yolov5/weights/MY.pt',
                        help='model.pt path(s)')
    parser.add_argument('--white_coat_yolo_weights', nargs='+', type=str, default='yolov5/weights/MY.pt',
                        help='model.pt path(s)')
    parser.add_argument('--glasses_yolo_weights', nargs='+', type=str, default='yolov5/weights/MY.pt',
                        help='model.pt path(s)')
    parser.add_argument('--gloves_yolo_weights', nargs='+', type=str, default='yolov5/weights/MY.pt',
                        help='model.pt path(s)')
    parser.add_argument('--face_mask_yolo_weights', nargs='+', type=str, default='yolov5/weights/MY.pt',
                        help='model.pt path(s)')

    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='video/porton-videos-test.avi', help='source')

    # output folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    # parameter
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # device select
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # save root
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    # deepsort configs
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    # other parameter
    parser.add_argument("--FrameNumberDetections", type=str, default="50")
    parser.add_argument("--FrameOriginSize", type=str, default=(720, 576))
    args = parser.parse_args()

    # init queue pipeline
    FrameProcessTrackerQueue = Queue()
    FrameOrlTrackerQueue = Queue()
    DetectionsTrackerQueue = Queue()
    FrameProcessAlarmQueue = Queue()
    FrameOrlAlarmQueue = Queue()
    DetectionsAlarmQueue = Queue()
    DetectionsQueue = Queue()
    AlarmQueue = Queue()
    FrameProcessGetQueue = Queue()
    FrameOrlGetQueue = Queue()
    FrameProcessInfQueue = Queue()
    FrameOrlInfQueue = Queue()

    # init parameter
    device = select_device(args.device)
    imgsz = args.img_size

    # init parameter V
    FrameNum = args.FrameNumberDetections
    FrameOutSize = (640, 480)
    names = ['people', 'white_coat', 'glasses', 'gloves', 'face_mask']

    # get detector parameter and init detector model
    yolo_weights_people = args.people_yolo_weights
    yolo_weights_white_coat = args.white_coat_yolo_weights
    yolo_weights_glasses = args.glasses_yolo_weights
    yolo_weights_gloves = args.gloves_yolo_weights
    yolo_weights_face_mask = args.face_mask_yolo_weights

    model_people = attempt_load(yolo_weights_people, map_location=device)
    model_white_coat = attempt_load(yolo_weights_white_coat, map_location=device)
    model_glasses = attempt_load(yolo_weights_glasses, map_location=device)
    model_gloves = attempt_load(yolo_weights_gloves, map_location=device)
    model_face_mask = attempt_load(yolo_weights_face_mask, map_location=device)

    # get tracker parameter and init tracker model
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # accelerate
    half = device.type != 'cpu'
    if half:
        model_people.half()
        model_white_coat.half()
        model_glasses.half()
        model_gloves.half()
        model_face_mask.half()

    # Run inference
    if device.type != 'cpu':
        model_people(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_people.parameters())))  # run once
        model_white_coat(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_white_coat.parameters())))
        model_glasses(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_glasses.parameters())))
        model_gloves(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_gloves.parameters())))
        model_face_mask(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_face_mask.parameters())))

    # video data catch
    rtsp = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
    rtmp = 'rtmp://172.18.16.153:1935/live/home'
    # cap = cv2.VideoCapture(rtsp)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./all.avi')

    sizeStr = str(FrameOutSize[0]) + 'x' + str(FrameOutSize[1])

    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s',sizeStr,
               '-r', '25',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp]

    pipe = subprocess.Popen(command, shell=False , stdin=subprocess.PIPE)
    # video_capture(FrameOrlQueue1, FrameProcessQueue)
    # inference(FrameProcessQueue, FrameOrlQueue1, FrameOrlQueue2, JudgeQueue, DetectionsQueue)
    # AfterProcess(FrameOrlQueue2, DetectionsQueue, JudgeQueue)
    Thread(target=video_capture, args=(FrameOrlGetQueue,FrameProcessGetQueue)).start()
    Thread(target=Tracker, args=(FrameProcessGetQueue,FrameOrlGetQueue,FrameProcessInfQueue,FrameOrlInfQueue,DetectionsTrackerQueue)).start()
    Thread(target=AfterProcess, args=(FrameOrlAlarmQueue,DetectionsAlarmQueue,DetectionsQueue,AlarmQueue)).start()
    Thread(target=Alarm, args=(FrameProcessTrackerQueue,FrameOrlTrackerQueue,DetectionsTrackerQueue,FrameProcessAlarmQueue,FrameOrlAlarmQueue,DetectionsAlarmQueue,DetectionsQueue,AlarmQueue)).start()