import cv2
import time
import subprocess
from threading import Thread
import threading


class PushStream(object):
    def __init__(self, rtmp, INPUT_W=640, INPUT_H=640):
        command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s', str(INPUT_W) + 'x' + str(INPUT_H), '-r', '25', '-i', '-', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-f', 'flv', '-flvflags', 'no_duration_filesize',
                    rtmp]
        self.pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
        self.judge = True

    def core(self, queueP):
        image_raw = queueP.get()
        self.pipe.stdin.write(image_raw.tobytes())

    def destory(self):
        self.pipe.terminate()

    def push(self, queueP):
        while self.judge:
            self.core(queueP)
        self.destory()

    def stop(self):
        self.judge = False

# 多线程
class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

def CurrentTime():
    ctime = time.localtime(time.time())
    timex = str(ctime[3]) + str(ctime[4]) + str(ctime[5])
    return int(timex)

class list_labid():
    def __init__(self):
        #self.labid= ["Stream1","Stream2","Stream3", "Stream4","Stream5","Stream6","Stream7", "Stream8", "Stream9","Stream10","Stream11","Stream12", "Stream13", "Stream14", "Stream15"]
        self.labid = ["Stream1", "Stream2", "Stream3", "Stream4", "Stream5", "Stream6", "Stream7", "Stream8", "Stream9",
                      "Stream10"]
        self.now = {}
        self.func = {}

    def delete_elements(self, num):
        string = str(num)
        off = self.now.pop(string)
        self.labid.append(off)

    def add_elements(self, num):
        if not str(num) in self.now.keys():
            string = str(num)
            stream_name = self.labid.pop()
            self.now[string] = stream_name
        else:
            stream = self.utlize(num)
            stream.stop_thread()
            self.del_func(num)
            self.delete_elements(num)

            string = str(num)
            stream_name = self.labid.pop()
            self.now[string] = stream_name

    def add_func(self, num, func):
        self.func[str(num)] = func

    def del_func(self, num):
        self.func.pop(str(num))

    def utlize1(self, num):
        return self.now[str(num)]

    def utlize(self, num):
        return self.func[str(num)]