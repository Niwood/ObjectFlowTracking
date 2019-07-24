
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

def sand():
    ld = {2:'a', 3:'dwa'}
    print(ld)
    ld.pop(min(ld), None)
    print(ld)
sand()
