'''

Geofence Component
Define the geofence

'''
import time
from utils import selectinwindow
import cv2
from PIL import Image
import numpy as np


def pre_defined_rog(vid, img_size):
    ret, frame = vid.read()
    frame = cv2.resize(frame, (720, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pilimg = Image.fromarray(frame)
    pilimg.thumbnail((img_size,img_size), Image.ANTIALIAS)

    img_ = np.array(pilimg)
    img = cv2.resize(img_, (frame.shape[1], frame.shape[0]))
    rog_area = [(int(img.shape[1]/2)-80,0), (int(img.shape[1]/2)+80,img.shape[1])]
    return rog_area

def shape_selector(video_path, vid):
    # Define the drag object
    rectI = selectinwindow.dragRect

    windowName = "Select ROG"
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

    if video_path=='cam':
        time.sleep(3)
    ret, image = vid.read()
    assert ret, 'Unable to capture video'
    image = cv2.resize(image, (720, 480))
    selectinwindow.init(rectI, image, windowName, image.shape[1], image.shape[0])
    cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
    # cv2.resizeWindow('Live', 800,600)

    # keep looping until rectangle finalized
    while True:
        # display the image
        cv2.imshow(windowName, rectI.image)
        key = cv2.waitKey(1) & 0xFF

        # if returnflag is True, break from the loop
        if rectI.returnflag == True:
            break

    cv2.destroyAllWindows()
    return [(rectI.outRect.x , rectI.outRect.y), (rectI.outRect.x+rectI.outRect.w , rectI.outRect.y+rectI.outRect.h)]
