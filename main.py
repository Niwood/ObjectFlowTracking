# OS
import os, sys, time, datetime, random, gc

# Utilities
from utils import utils
from utils.models import *
from utils.sort import *

# Modules
from fpm.perception_logic_component import *
from fpm.geofence_component import *
from otm.object_recognition_component import *
from otm.image_processing_component import *

# PyTorch
import torch

# CV tools
import cv2
from PIL import Image


'''
Configuration
'''
video_path = 'images/blueangels.jpg' # Set parameter to "cam" for webcam
video_path = 'cam'
save_record = False
enable_ROG = False
manual_ROG_selection = False # Pre defined ROG area in main loop
enable_otm = True

yolo_model = 'default' # default or tiny
# img_size = 416
# img_size = 256
img_size = 160
conf_thres = 0.7
nms_thres = 0.4
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
class_path='config/coco.names'



'''
Classifier configuration and weights
'''
if yolo_model == 'tiny':
    weights_path='config/yolov3-tiny.weights'
    if img_size == 416:
        config_path='config/yolov3-tiny-416.cfg'
    elif img_size == 256:
        config_path='config/yolov3-tiny-256.cfg'
    elif img_size == 160:
        config_path='config/yolov3-tiny-160.cfg'
elif yolo_model == 'default':
    weights_path='config/yolov3.weights'
    if img_size == 416:
        config_path='config/yolov3-416.cfg'
    elif img_size == 256:
        config_path='config/yolov3-256.cfg'
    elif img_size == 160:
        config_path='config/yolov3-160.cfg'



'''
Load model and set to evaluation mode
'''
CUDA = torch.cuda.is_available()
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
if CUDA:
    model.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
model.eval()
classes = utils.load_classes(class_path)



'''
Video source and motion tracker
'''
mot_tracker = Sort()
if video_path=='cam':
    vid = cv2.VideoCapture(0)
else:
    vid = cv2.VideoCapture(video_path)



'''
Shape selection of ROG
'''
if enable_ROG:
    if manual_ROG_selection:
        rog_area = shape_selector(video_path, vid)
    else:
        rog_area = pre_defined_rog(vid, img_size)
elif not enable_ROG:
    rog_area = [(0, 0), (0, 0)]



'''
Main loop
'''
if save_record:
    out_video = cv2.VideoWriter('images/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (720, 480))
obj_list = {}
rog_count = 0
rog_count_tot = 0
frames = 0
starttime = time.time()


while(True):


    ''' Pre-process image '''
    pilimg, frame, frame_orig, ret, img = pre_process(vid, img_size)
    if not ret:
        print('End of video.')
        break


    ''' Image Detection '''
    if enable_otm:
        detections = detect_image(pilimg, img_size, conf_thres, nms_thres, model, Tensor)
    else:
        detections = None

    frames += 1
    rog_count = 0
    cv2.rectangle(frame, rog_area[0], rog_area[1],(0,20,255),-1)
    alpha = 0.4
    cv2.addWeighted(frame, alpha, frame_orig, 1 - alpha, 0, frame)


    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x


    obj_list = {}
    if detections is not None:

        tracked_objects = mot_tracker.update(detections.cpu())
        for i in [int(i[4]) for i in tracked_objects]:
            obj_list[i] = []

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:


            if True:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                x_point = int((2*x1+box_w)/2)
                y_point = int(y1+box_h)
                obj_list[obj_id] = [x_point ,y_point]

                color = colors[int(obj_id) % len(colors)]
                # color = (255, 153, 255)
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                # cv2.circle(frame, (x_point,y_point), 10, color, thickness=-1, lineType=8, shift=0)
                # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x_point, y_point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                ''' Count objects in ROG '''
                rog_count = region_of_geofence(obj_list, rog_area)

    else:
        detections = []



    ''' Info panel '''
    cv2.rectangle(frame, (0, 0), (180, 70), (0,0,0), -1)
    cv2.putText(frame, str(round(frames / (time.time() - starttime),2))+' FPS', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(frame, 'OBJECTS FOUND: '+str(len(detections)), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(frame, 'INSIDE ROG: '+str(rog_count), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)



    ''' Display image '''
    cv2.imshow('Live', frame)
    if save_record:
        out_video.write(frame)

    gc.collect()
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break


print('Total execution time:',time.time()-starttime)
cv2.destroyAllWindows()
if save_record:
    out.release()
print('--- END OF PROCESS ---')
