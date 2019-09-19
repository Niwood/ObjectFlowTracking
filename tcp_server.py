import socket
import numpy as np
import pickle
import cv2
import time


class TCP_Server:

    def __init__(self, HOST=None, PORT=None):
        self.HOST = HOST
        self.PORT = PORT
        if self.HOST and self.PORT:
            self.server, self.conn = self.start_server()

        # Recieve short message for packet size and iterations
        self.recieve_shortMSG()

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((socket.gethostname(), self.PORT))
        server.listen()
        print('Server',socket.gethostbyname(socket.gethostname()),'listening on PORT', self.PORT)
        conn, addr = server.accept()
        print('Connected to', addr)
        return server, conn

    def recieve_shortMSG(self):
        packet_from_client = self.conn.recv(4096)
        self.SM_packet = pickle.loads(packet_from_client)

        # Send confirmation to client
        self.send_confirmation()

    def recieve_longMSG(self, size, iterations):
        # Loop through all iterations specified by the client
        counter = 1
        while True:
            packet_from_client = b''
            size_left = size

            # Loop through all recieved messages from the client until complete size has been recieved
            while True:
                data = self.conn.recv(size_left)
                packet_from_client += data
                size_left -= len(data)
                # print(len(packet_from_client))

                if len(packet_from_client) >= size:
                    break
            LM_packet = pickle.loads(packet_from_client)
            did = self.DID_interpreter(LM_packet['DID'])

            # Show stream
            cv2.imshow('frame',LM_packet['PAYLOAD'])
            cv2.waitKey(1)

            # Send confirmation to client
            self.send_confirmation()

            # Count number of iterations
            counter += 1
            if counter == iterations: break

    def recieve_single_longMSG(self, size, iterations):
        # Loop through all iterations specified by the client
        packet_from_client = b''
        size_left = size

        # Loop through all recieved messages from the client until complete size has been recieved
        while True:
            data = self.conn.recv(size_left)
            packet_from_client += data
            size_left -= len(data)
            # print(len(packet_from_client))

            if len(packet_from_client) >= size:
                break
        LM_packet = pickle.loads(packet_from_client)
        did = self.DID_interpreter(LM_packet['DID'])

        # Send confirmation to client
        # self.send_confirmation()
        return LM_packet['PAYLOAD']

    def send_confirmation(self):
        msg = "I got the short msg. /SERVER"
        self.conn.send(pickle.dumps(msg))

    def DID_interpreter(self, DID):
        if DID == 'frame':
            return
        elif DID == 'end_process':
            return 'end_process'

    def read(self):
        return self.recieve_single_longMSG(self.SM_packet['PAYLOAD'], self.SM_packet['ITERATIONS'])

    def close_server(self):
        try:
            self.server.close()
            print('Server closed.')
        except:
            print('Could not close the server properly.')
            quit()


if __name__ == "__main__":
    # HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    # Create the connection object and connect to server
    connection = TCP_Server(HOST='10.0.0.9', PORT=65433)

    pause = 3
    while True:
        frame = connection.read()
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        for i in range(pause):
            print(i)
            time.sleep(1)
        connection.send_confirmation()

    # Close the server
    connection.close_server()
