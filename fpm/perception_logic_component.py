'''

Perception Logic Component
Counting logic

'''

'''
Object in ROG counter
'''
def region_of_geofence(obj_list, rog_area):
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
