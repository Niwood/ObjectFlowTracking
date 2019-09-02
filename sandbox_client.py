import socket
import numpy as np
import cv2
import pickle
import sys


class TCP_client:

    def __init__(self, HOST, PORT, ITERATIONS):
        self.HOST = HOST
        self.PORT = PORT
        self.ITERATIONS = ITERATIONS
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()
        if sys.platform == 'darwin':
            self.cam = cv2.VideoCapture(0)
        elif: sys.platform == 'linux':
            self.cam = jetson_camera()

    def connect_to_server(self):
        try:
            self.client.connect((self.HOST, self.PORT))
            print('Connection to server established')
        except:
            print('Could not connect to server on', self.HOST, 'PORT:',self.PORT)
            quit()

    def send_shortMSG(self, type = None):
        if type == 'packet_size':
            sample_packet = self.send_longMSG(get_packet_size = True)
            sample_packet_size = len(pickle.dumps(sample_packet))
            # print('Packet size to be sent:',sample_packet_size)
            packet = {'DID':'packet_size', 'ITERATIONS':self.ITERATIONS, 'PAYLOAD':sample_packet_size}
            self.client.send(pickle.dumps(packet))
            self.recieve_from_server(supression = True)
        elif type == 'end_process':
            packet = {'DID':'end_process','PAYLOAD':None}
            self.client.send(pickle.dumps(packet))
        else:
            print('Unknown type for short MSG.')
            self.close_client()

    def send_longMSG(self, get_packet_size = False):
        if get_packet_size:
            payload = self.generate_payload()
            packet = {'DID':'frame', 'PAYLOAD':payload}
            return packet
        else:
            k = 0
            while True:
                payload = self.generate_payload()
                packet = {'DID':'frame','PAYLOAD':payload}
                packet_pickled = pickle.dumps(packet)
                # print('Size of package to be sent:',len(packet_pickled))
                self.client.send(packet_pickled)
                k += 1
                if k == self.ITERATIONS: break

    def generate_payload(self):
        ret, frame = self.cam.read()
        return frame

    def recieve_from_server(self, supression = True):
        packet_from_server = self.client.recv(4096)
        if not supression:
            print(pickle.loads(packet_from_server))

    def gstreamer_pipeline (self, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
        return ('nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), '
        'width=(int)%d, height=(int)%d, '
        'format=(string)NV12, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

    def jetson_camera(self):
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        cam = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if cam.isOpened():
            # window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
            # while cv2.getWindowProperty('CSI Camera',0) >= 0:
                # ret_val, img = cam.read();
                # cv2.imshow('CSI Camera',img)
                # keyCode = cv2.waitKey(30) & 0xff
                # if keyCode == 27:
                   # break
            # cam.release()
            # cv2.destroyAllWindows()
            return cam
        else:
            print('Unable to open camera. Implementation for Jetson Nano.')
            quit()

    def close_client(self):
        try:
            self.client.close()
            self.cam.release()
            print('Connection to server closed')
        except:
            print('Could not close the client properly.')
            quit()


# Specify the HOST and PORT used by server
HOST = '127.0.0.1'
PORT = 65430

# Specify how many frames the client should send
ITERATIONS = 10

# Create the connection object and connect to server
connection = TCP_client(HOST, PORT, ITERATIONS)

# Send SM to server specifying the packet size and iterations
connection.send_shortMSG(type = 'packet_size')

# Start LM communication with server, specifying number of iterations
connection.send_longMSG(get_packet_size = False)

# Close the client
connection.close_client()
