import socket
import numpy as np
import pickle
import cv2
import time

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65430        # Port to listen on (non-privileged ports are > 1023)


def recieve_longMSG(conn, size):
    packet_from_client = b''
    while True:
        data = conn.recv(size)
        packet_from_client += data
        if len(packet_from_client) == size: break
    packet = pickle.loads(packet_from_client)
    return packet


def recieve_shortMSG(conn):
    packet_from_client = conn.recv(4096)
    # print('Packet size from client:',len(packet_from_client))
    packet_size = pickle.loads(packet_from_client)
    return packet_size


def send_confirmation(conn):
    msg = "I got the short msg. /SERVER"
    conn.send(pickle.dumps(msg))


def DID_interpreter(DID):
    if DID == 'frame':
        return
    elif DID == 'end_process':
        return 'end_process'


def start_server():
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind((HOST, PORT))
    serv.listen()
    print('Server listening on PORT', PORT)
    conn, addr = serv.accept()
    print('Connected to', addr)
    # Recieve short message for packet size
    packet_size = recieve_shortMSG(conn)
    send_confirmation(conn)

    i = 0
    while True:
        packet = recieve_longMSG(conn, packet_size['PAYLOAD'])

        did = DID_interpreter(packet['DID'])
        print(did)
        if did == 'end_process':
            print('Recieved end_process from client')
            send_confirmation(conn)
            break

        # Show stream
        cv2.imshow('frame',packet['PAYLOAD'])
        cv2.waitKey(1)

        send_confirmation(conn)

    # Close server
    serv.close()
    cv2.destroyAllWindows()
    print('SERVER CLOSED')


def test_longMSG():
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind((HOST, PORT))
    serv.listen()
    conn, addr = serv.accept()

    i = 0
    while True:

        from_client = b''
        while True:

            data = conn.recv(4096)
            from_client += data
            if len(from_client) == 80159: break

        print('MSG size from client',len(from_client))

        # Send confirmation msg to client
        msg = "This was sent from the SERVER"
        conn.send(pickle.dumps(msg))

        i += 1
        if i==2:
            break

    # Close server
    conn.close()
    print('client disconnected')

def test_shortMSG():
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind((HOST, PORT))
    serv.listen()
    conn, addr = serv.accept()

    from_client = conn.recv(4096)
    print('Msg from client:',pickle.loads(from_client))
    msg = "I got the short msg. /SERVER"
    conn.send(pickle.dumps(msg))

    conn.close()
    print('client disconnected')

start_server()
