'''

Object Recognition Component
Define object placement in image

'''

import torch
from torchvision import transforms
from torch.autograd import Variable
from utils import utils

# Profiler
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

'''
Image detection Function
'''
@profile
def detect_image(img, img_size, conf_thres, nms_thres, model, Tensor):

    ''' Scale and pad image '''
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),(128,128,128)),
         transforms.ToTensor(),
         ])

    ''' Convert image to Tensor '''
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    ''' Run inference on the model and get detections '''
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]
