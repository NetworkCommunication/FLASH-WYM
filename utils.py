import numpy as np
import socket

class trafficCompression:
    def __init__(self):
        test = 1

    def check_resp(self, conn):
        resp = conn.recv(1).decode('utf8')
        if resp != '/':
            raise RuntimeError('Receiver did not respond to data send.')

    def send_ack(self, conn):
        conn.send(bytes('/', 'utf8'))

    def read_header(self, conn):
        header = ""
        char = ""
        while char != "|":
            char = conn.recv(1).decode('utf8')
            header += char
        return header[:-1]

    def send_array_over_socket(self, conn, array):
        data = array.tobytes()
        shape = array.shape

        header = str(len(data)) + ','
        header += str(array.dtype) + ','
        for s in shape:
            header += str(s) + ','
        header = header[:-1] + "|"
        header = bytes(header, 'utf8')

        conn.send(header)
        self.check_resp(conn)
        conn.send(data)
        self.check_resp(conn)

    def recv_array_from_socket(self, conn):
        header = self.read_header(conn)
        if "STOP" in header:
            self.send_ack(conn)
            return "STOP"

        vals = header.split(',')
        data_len = int(vals[0])
        dtype = vals[1]
        shape = [int(n) for n in vals[2:]]
        data = np.empty(shape, dtype=dtype)
        self.send_ack(conn)
        p = conn.recv_into(data, data_len, socket.MSG_WAITALL)
        self.send_ack(conn)

        return data

    def send_num_over_socket(self, conn, n):
        chars = str(n)
        for c in chars:
            conn.send(bytes(c, 'utf8'))
        conn.send(bytes('|', 'utf8'))
        self.check_resp(conn)

    def recv_num_from_socket(self, conn):
        n = self.read_header(conn)
        self.send_ack(conn)
        return n

    def send_int_over_socket(self, conn, i):
        self.send_num_over_socket(conn, i)

    def recv_int_from_socket(self, conn):
        return int(self.recv_num_from_socket(conn))

    def send_float_over_socket(self, conn, f):
        self.send_num_over_socket(conn, f)

    def recv_float_from_socket(self, conn):
        return float(self.recv_num_from_socket(conn))

    def send_str_over_socket(self, conn, s):
        conn.send(bytes(s + '|', 'utf8'))
        self.check_resp(conn)

    def recv_str_from_socket(self, conn):
        s = self.read_header(conn)
        self.send_ack(conn)
        return s
