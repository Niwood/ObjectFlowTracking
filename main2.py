# OS
import os, sys, time, datetime, random, gc

# Utilities
from utils.utils import *
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

# TCP Server
from tcp_server import TCP_Server

# Profiler
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


class ObjectFlowTracking():

    def __init__(self):
        '''
        Configuration
        '''
        # self.video_path = 'images/traffic3.mp4' # Set parameter to "cam" for webcam
        # self.video_path = 'images/blueangels.jpg'
        # self.video_path = 'cam'
        self.video_path = 'client'
        self.save_record = False
        self.enable_ROG = False
        self.manual_ROG_selection = False # Pre defined ROG area in main loop
        self.enable_otm = True
        self.print_tracking = True

        self.yolo_model = 'default' # default or tiny
        # self.img_size = 416
        self.img_size = 256
        # img_size = 160
        self.conf_thres = 0.7
        self.nms_thres = 0.4
        self.colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
        self.class_path='config/coco.names'

        '''
        TCP Server
        '''
        try:
            self.connection = TCP_Server(HOST='10.0.0.9', PORT=65433)
        except:
            print('Could not connect to client')
            quit()

        '''
        Classifier configuration and weights
        '''
        edit_cfg_resolution(self.img_size, self.yolo_model)
        if self.yolo_model == 'tiny':
            self.weights_path='config/yolov3-tiny.weights'
            self.config_path='config/yolov3-tiny.cfg'
        elif self.yolo_model == 'default':
            self.weights_path='config/yolov3.weights'
            self.config_path='config/yolov3.cfg'

        '''
        Load model and set to evaluation mode
        '''
        CUDA = torch.cuda.is_available()
        self.model = Darknet(self.config_path, img_size = self.img_size)
        self.model.load_weights(self.weights_path)
        if CUDA:
            self.model.cuda()
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor
        self.model.eval()
        self.classes = load_classes(self.class_path)

        '''
        Video source and motion tracker
        '''
        self.mot_tracker = Sort()
        if self.video_path=='cam':
            self.vid = cv2.VideoCapture(0)
            self.number_of_frames = 1
        elif self.video_path=='client':
            # self.vid = cv2.VideoCapture(0)
            self.number_of_frames = 1
        else:
            self.vid = cv2.VideoCapture(self.video_path)
            self.number_of_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

        '''
        Shape selection of ROG
        '''
        if self.enable_ROG:
            if self.manual_ROG_selection:
                self.rog_area = shape_selector(self.video_path, self.vid)
            else:
                self.rog_area = pre_defined_rog(self.vid, self.img_size)
        elif not self.enable_ROG:
            self.rog_area = [(0, 0), (0, 0)]

        '''
        Run image detection
        '''
        self.run()


    def run(self):
        if self.save_record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter('images/output.mp4',fourcc, 20.0, (int(vid.get(3)),int(self.vid.get(4))))
        obj_list = {}
        rog_count = 0
        rog_count_tot = 0
        frames = 0
        starttime = time.time()

        '''
        Main loop
        '''
        while True:

            ''' Grab frame from server '''
            frame = self.connection.read()
            # frame = cv2.flip(frame,-1)

            ''' Pre-process image '''
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pilimg = Image.fromarray(frame)
            pilimg.thumbnail((self.img_size,self.img_size), Image.ANTIALIAS)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output = frame.copy()
            frame_orig = frame.copy()
            img_ = np.array(pilimg)
            img = cv2.resize(img_, (frame.shape[1], frame.shape[0]))

            # if not ret:
            #     print('End of video.')
            #     break


            ''' Image Detection, shape: (x1, y1, x2, y2, object_conf, class_score, class_pred) '''
            if self.enable_otm:
                detections = detect_image(pilimg, self.img_size, self.conf_thres, self.nms_thres, self.model, self.Tensor)
            else:
                detections = None

            frames += 1
            rog_count = 0
            cv2.rectangle(frame, self.rog_area[0], self.rog_area[1],(0,20,255),-1)
            alpha = 0.4
            cv2.addWeighted(frame, alpha, frame_orig, 1 - alpha, 0, frame)

            pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x

            if detections is not None:

                tracked_objects = self.mot_tracker.update(detections.cpu())
                tracked_object_list = [int(i[4]) for i in tracked_objects]

                # Remove objects when more than 100 objects
                if len(obj_list)>1000:
                    obj_list.pop(min(obj_list), None)

                # Add newly detected objects
                for i in tracked_object_list:
                    if i not in obj_list:
                        obj_list[i] = {'class':'','position':[]}

                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)

                for track_obj, obj_conf in zip(tracked_objects,list(detections[:, 4].numpy())):

                    x1, y1, x2, y2, obj_id, cls_pred = track_obj
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    x_point = int((2*x1+box_w)/2)
                    y_point = int(y1+box_h)
                    color = self.colors[int(obj_id) % len(self.colors)]
                    # color = (255, 153, 255)
                    cls = self.classes[int(cls_pred)]
                    obj_list[obj_id]['class'] = cls
                    obj_list[obj_id]['position'].append((x_point ,y_point))

                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 1)
                    cv2.putText(frame, cls.capitalize() + ":" + str(int(obj_id)) + '['+ str(int(obj_conf*100)) +'%]', (x_point+10, y_point-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1,cv2.LINE_AA)

                    if self.print_tracking:
                        for circ in obj_list[obj_id]['position']:
                            cv2.circle(frame, circ, 3, color, thickness=-1, lineType=8, shift=0)
                    else:
                        cv2.circle(frame, obj_list[obj_id]['position'][-1], 3, color, thickness=-1, lineType=8, shift=0)

                    ''' Count objects in ROG '''
                    # rog_count = region_of_geofence(obj_list, rog_area)
                    rog_count = 0

            else:
                detections = []


            ''' Info panel '''
            cv2.rectangle(frame, (0, 0), (180, 70), (0,0,0), -1)
            cv2.putText(frame, str(round(frames / (time.time() - starttime),2))+' FPS', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            cv2.putText(frame, 'OBJECTS FOUND: '+str(len(detections)), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            # cv2.putText(frame, 'INSIDE ROG: '+str(rog_count), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            cv2.putText(frame, 'Frames left: '+str(self.number_of_frames-frames), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)


            ''' Display image '''
            cv2.imshow('Live', frame)
            if self.save_record:
                out_video.write(frame)

            gc.collect()
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                # self.connection.close_server()
                break

            ''' Send delivery confirmation to server '''
            self.connection.send_confirmation()

ObjectFlowTracking()
