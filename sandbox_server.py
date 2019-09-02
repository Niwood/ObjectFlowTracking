import socket
import numpy as np
import pickle
import cv2
import time


class TCP_server:

    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.server, self.conn = self.start_server()

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.HOST, self.PORT))
        server.listen()
        print('Server listening on PORT', PORT)
        conn, addr = server.accept()
        print('Connected to', addr)
        return server, conn

    def recieve_shortMSG(self):
        packet_from_client = self.conn.recv(4096)
        SM_packet = pickle.loads(packet_from_client)
        # print('Packet to be recieved:',SM_packet['PAYLOAD'])

        # Send confirmation to client
        self.send_confirmation()
        return SM_packet

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
            did = DID_interpreter(LM_packet['DID'])

            # Show stream
            cv2.imshow('frame',LM_packet['PAYLOAD'])
            cv2.waitKey(1)

            # Send confirmation to client
            self.send_confirmation()

            # Count number of iterations
            counter += 1
            if counter == iterations: break

    def send_confirmation(self):
        msg = "I got the short msg. /SERVER"
        self.conn.send(pickle.dumps(msg))

    def DID_interpreter(self, DID):
        if DID == 'frame':
            return
        elif DID == 'end_process':
            return 'end_process'

    def close_server(self):
        try:
            self.server.close()
            print('Server closed.')
        except:
            print('Could not close the server properly.')
            quit()


# Specify the HOST and PORT used by server
# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
HOST = '10.0.0.9'
PORT = 65430        # Port to listen on (non-privileged ports are > 1023)

# Create the connection object and connect to server
connection = TCP_server(HOST, PORT)

# Recieve short message for packet size and iterations
SM_packet = connection.recieve_shortMSG()

# Recieve LM from client
connection.recieve_longMSG(SM_packet['PAYLOAD'], SM_packet['ITERATIONS'])

# Close the server
connection.close_server()
