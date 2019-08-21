import socket
import numpy as np
import cv2
import pickle
import sys


class TCP_client:

    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()
        self.cam = cv2.VideoCapture(0)

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
            packet = {'DID':'packet_size','PAYLOAD':sample_packet_size}
            self.client.send(pickle.dumps(packet))
        elif type == 'end_process':
            packet = {'DID':'end_process','PAYLOAD':None}
            self.client.send(pickle.dumps(packet))
        else:
            print('Unknown type for short MSG.')
            self.close_client()

    def send_longMSG(self, get_packet_size = False, iterations = None):
        if get_packet_size:
            payload = self.generate_payload()
            packet = {'DID':'frame', 'PAYLOAD':payload}
            return packet
        else:
            k = 0
            while True:
                payload = self.generate_payload()
                packet = {'DID':'frame', 'PAYLOAD':payload}
                self.client.send(pickle.dumps(packet))
                k += 1
                if k == iterations:
                    break

    def generate_payload(self):
        ret, frame = self.cam.read()
        return frame

    def recieve_from_server(self, supression = True):
        packet_from_server = self.client.recv(4096)
        if not supression:
            print(pickle.loads(packet_from_server))

    def close_client(self):
        try:
            self.client.close()
            print('Connection to server closed')
        except:
            print('Could not properly close client')
            quit()


# Specify the HOST and PORT used by server
HOST = '127.0.0.1'
PORT = 65430

# Create the connection object and connect to server
connection = TCP_client(HOST, PORT)

# Send SM to server specifying the packet size
connection.send_shortMSG(type = 'packet_size')

# Recieve handshake from server
connection.recieve_from_server(supression = False)

# Start LM communication with server, specifying number of iterations
connection.send_longMSG(get_packet_size = False, iterations = 100)

# Close the client
connection.close_client()
