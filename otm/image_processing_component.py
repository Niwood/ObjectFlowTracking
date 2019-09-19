
import cv2
from PIL import Image
import numpy as np

# Profiler
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

def pre_process(vid, img_size):
    ret, frame = vid.read()

    if not ret:
        return None, None, None, ret, None

    frame = cv2.flip( frame, 1 )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pilimg = Image.fromarray(frame)
    pilimg.thumbnail((img_size,img_size), Image.ANTIALIAS)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output = frame.copy()
    frame_orig = frame.copy()
    img_ = np.array(pilimg)
    img = cv2.resize(img_, (frame.shape[1], frame.shape[0]))

    return pilimg, frame, frame_orig, ret, img


def pre_processOLD(vid, img_size):
    ret, frame = vid.read()

    if not ret:
        return None, None, None, ret, None

    frame = cv2.flip( frame, 1 )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pilimg = Image.fromarray(frame)
    pilimg.thumbnail((img_size,img_size), Image.ANTIALIAS)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output = frame.copy()
    frame_orig = frame.copy()
    img_ = np.array(pilimg)
    img = cv2.resize(img_, (frame.shape[1], frame.shape[0]))

    return pilimg, frame, frame_orig, ret, img
