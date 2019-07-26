from models import *
from utils import *
import selectinwindow

import os, sys, time, datetime, random, gc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import cv2
from sort import *

from PIL import Image



'''
Parameters
'''
video_path = 'images/blueangels.jpg' # Set parameter to "cam" for webcam
save_record = False
manual_ROG_selection = False # Pre defined ROG area in main loop

# img_size = 416
# img_size = 256
img_size = 160
conf_thres = 0.7
nms_thres = 0.4
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'



'''
Load model and set to eval mode
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
Image detection Function
'''
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),(128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


'''
Object in ROG counter
'''
def region_of_geofence(obj_list):
    # dist = cv2.pointPolygonTest((rog_area[0], rog_area[1]),(x_point,y_point),True)
    # print('Dist: ',dist)

    x1 = rog_area[0][0]
    y1 = rog_area[0][1]
    x2 = rog_area[1][0]
    y2 = rog_area[1][1]

    rog_count = 0
    for obj in obj_list:
        if len(obj_list[obj])>0:
            if obj_list[obj][0]>x1 and obj_list[obj][0]<x2 and obj_list[obj][1]>y1 and obj_list[obj][1]<y2:
                rog_count += 1
    return rog_count



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
def shape_selector():
    # Define the drag object
    global rog_area
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

    rog_area = [(rectI.outRect.x , rectI.outRect.y), (rectI.outRect.x+rectI.outRect.w , rectI.outRect.y+rectI.outRect.h)]
    cv2.destroyAllWindows()

if manual_ROG_selection:
    shape_selector()




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
    ret, frame = vid.read()
    if not ret:
        print('End of video.')
        break
    frames += 1
    frame = cv2.resize(frame, (720, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    pilimg = Image.fromarray(frame)
    pilimg.thumbnail((img_size,img_size), Image.ANTIALIAS)
    detections = detect_image(pilimg)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output = frame.copy()
    frame_orig = frame.copy()
    img_ = np.array(pilimg)
    img = cv2.resize(img_, (frame.shape[1], frame.shape[0]))


    # Define ROG area
    if not manual_ROG_selection:
        rog_area = [(int(img.shape[1]/2)-80,0), (int(img.shape[1]/2)+80,img.shape[1])]
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

            # if int(cls_pred) == 2: # Specify which classes to include
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

                # Count objects in ROG
                rog_count = region_of_geofence(obj_list)

    else:
        detections = []


    # Info panel
    cv2.rectangle(frame, (0, 0), (180, 70), (0,0,0), -1)
    cv2.putText(frame, str(round(frames / (time.time() - starttime),2))+' FPS', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(frame, 'OBJECTS FOUND: '+str(len(detections)), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(frame, 'INSIDE ROG: '+str(rog_count), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    # Show image
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
