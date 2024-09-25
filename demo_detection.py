import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import numpy as np
import cv2
import threading
import random
from copy import deepcopy


def plot_one_box(x, img, label=None, line_thickness=None):
    tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    color = [0,0,255]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)

class YoLov5TRT(object):
    def __init__(self, PLUGIN_LIBRARY, engine_file_path, H, W, categories, cameraNum=1):
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.cfx = cuda.Device(0).make_context()
        self.categories = [categories]
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        self.INPUT_W = W
        self.INPUT_H = H
        self.CONF_THRESH = 0.1
        self.IOU_THRESHOLD = 0.4

        for _ in list(range(cameraNum)):
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(cuda_mem))
                if engine.binding_is_input(binding):
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        self.cfx.push()
        input_image1, image_raw1, origin_h1, origin_w1 = self.preprocess_image(input_image_path)
        np.copyto(host_inputs[0], input_image1.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings[0:2], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.cfx.pop()
        output1 = host_outputs[0]
        result_boxes1, result_scores1, result_classid1 = self.post_process(output1, origin_h1, origin_w1)
        for i in range(len(result_boxes1)):
            box = result_boxes1[i]
            plot_one_box(box, image_raw1, label="{}:{:.2f}".format(self.categories[int(result_classid1[i])], result_scores1[i]), )
        return image_raw1, result_boxes1, result_scores1, result_classid1

    def destroy(self):
        self.cfx.pop()

    def preprocess_image(self, image_raw):
        h, w, c = image_raw.shape
        image = image_raw
        r_w = self.INPUT_W / w
        r_h = self.INPUT_H / h
        if r_h > r_w:
            tw = self.INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.INPUT_H - th) / 2)
            ty2 = self.INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = self.INPUT_H
            tx1 = int((self.INPUT_W - tw) / 2)
            tx2 = self.INPUT_W - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.INPUT_W / origin_w
        r_h = self.INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        pred = torch.Tensor(pred).cuda()
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]
        si = scores > self.CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid


def main(mode, video_root):
    import os
    save_root = './visual'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = os.path.join(save_root, mode)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dict_para = {
        'people':(PLUGIN_LIBRARY, engine_file_path, 640, 640),
        'coat': (PLUGIN_LIBRARY, engine_file_path, 640, 640),
        'glasses':(PLUGIN_LIBRARY, engine_file_path, 1280, 720),
        'gloves':(PLUGIN_LIBRARY, engine_file_path, 1280, 720),
                 }

    PLUGIN_LIBRARY, engine_file_path, H, W = dict_para[mode]
    model = YoLov5TRT(PLUGIN_LIBRARY, engine_file_path, H, W, mode)

    cap = cv2.VideoCapture(video_root)
    c = 0
    while cap.isOpened():
        _, frame = cap.read()
        pic = model.infer(frame)
        filename = os.path.join(save_path, 'frame' + str(c) + '.jpg')
        cv2.imwrite(filename, pic)


if __name__ == '__main__':
    mode = 'people'
    video_root = 'test.mp4'
    main(mode, video_root)