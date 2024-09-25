import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import ctypes
import numpy as np
import cv2
import torch
import torchvision
import threading


class YoLov5TRTN(object):
    def __init__(self, PLUGIN_LIBRARY, engine_file_path):
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.cfx = cuda.Device(0).make_context()
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

        self.INPUT_W = 640
        self.INPUT_H = 640
        self.CONF_THRESH = 0.1
        self.IOU_THRESHOLD = 0.4

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, image):
        threading.Thread.__init__(self)
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
        np.copyto(host_inputs[0], input_image.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.cfx.pop()
        output = host_outputs[0]
        result_boxes = self.post_process(output, origin_h, origin_w)
        return result_boxes

    def destroy(self):
        self.cfx.pop()

    def preprocess_image(self, image_raw ):
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
        si = scores > self.CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        return result_boxes