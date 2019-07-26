
def drag():
    import cv2 # Opencv ver 3.1.0 used
    import numpy as np
    import time

    import sys
    # Set recursion limit
    sys.setrecursionlimit(10 ** 9)

    import selectinwindow

    # Define the drag object
    rectI = selectinwindow.dragRect

    # Initialize the  drag object
    wName = "select region"


    # vid = cv2.VideoCapture(0)
    # time.sleep(3)
    # ret, image = vid.read()

    image = cv2.imread('images/blueangels.jpg')
    selectinwindow.init(rectI, image, wName, image.shape[1], image.shape[0])

    cv2.namedWindow(rectI.wname)
    cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)

    # keep looping until rectangle finalized
    count = 0
    while True:
        count += 1
        print(count, rectI.returnflag)
        # display the image
        cv2.imshow(wName, rectI.image)
        key = cv2.waitKey(1) & 0xFF

        # if returnflag is True, break from the loop
        if rectI.returnflag == True:
            print('ENDED')
            break

        # ch = cv2.waitKey(1) & 0xFF
        # if ch == 27:
        #     break

    print("Dragged rectangle coordinates")
    print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y))

    # close all open windows
    cv2.destroyAllWindows()

def configa():
    from utils import utils
    edit_cfg_resolution(15, 15)

def edit_cfg_resolution(width, height):
    config_path='config/yolov3-416.cfg'
    config_file = open(config_path, "r")
    all_rows = []
    for row in config_file:
        if 'width' in row:
            all_rows.append('width='+str(width)+'\n')
            continue
        elif 'height' in row:
            all_rows.append('height='+str(height)+'\n')
            continue
        all_rows.append(row)
    config_file.close()

    config_file = open(config_path, "w")
    for row in all_rows:
        config_file.write(row)
    config_file.close()

def sand():
    ld = {2:'a', 3:'dwa'}
    print(ld)
    ld.pop(min(ld), None)
    print(ld)

configa()
