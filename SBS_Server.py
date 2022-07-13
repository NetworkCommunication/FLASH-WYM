import select
import socket
import struct
import pickle
from collections import Counter
import tensorflow as tf
import numpy as np
import argparse
import config
import protocol
import random
from VAE import ANNVAE
from utils import *
from VAE import vae_loss
from numpy.random import random, randint
from itertools import groupby
from threading import Thread


client_num = config.client_num
top_num = config.top_num
round_num = config.round_num
compress_level = config.compress_level

class ModelAggregate:
    def __init__(self):
        np.random.seed(1337)  # for reproducibility
        self.initNetwork()

    def initNetwork(self):
        self.vae_model = ANNVAE()
        input_data = tf.zeros((1,3883))
        self.vae_model(input_data)

    def setNetwork(self, weight):
        self.vae_model.set_weights(weight)

    def getNetwork(self):
        weight = self.vae_model.get_weights()
        return weight

    def countAcc(self,send_from_client):
        rec_list=[]
        temp=[]
        for li in send_from_client:
            for ji in li:
                temp.append(ji)
        print('Before counter', len(temp))
        Ranking = Counter(temp)
        Ranking_top = Ranking.most_common(client_num*top_num)
        for i in Ranking_top:
            rec_list.append(int(i[0]))
        print('After counter:',len(rec_list))

        self.sbs_rec_list = rec_list[0:top_num]
        print('sbs rec list:', len(self.sbs_rec_list))

class DataCompression:
    def __init__(self):
        self.traff_comp = trafficCompression()

    # quantization step
    def quantize(self, t_in, s):
        m = (np.amax(np.absolute(t_in)) * s).astype(np.float32)
        return np.around(t_in / m, decimals=0).astype(np.int8), m

    # dequantization step
    def dequantize(self, t_q, m):
        return m * t_q

    # quartic encoding
    def q_encode(self, t_q):
        t_flat = (t_q + 1).astype(np.uint8).flatten()
        n_0s = int(np.ceil(len(t_flat)/5.0)*5.0) - len(t_flat)
        p = np.split(np.pad(t_flat, (0, n_0s), 'constant'), 5)
        a = (p[0]*81) + (p[1]*27) + (p[2]*9) + (p[3]*3) + p[4]
        return a

    # quartic decoding
    def q_decode(self, a, shape):
        ps = [np.floor(np.mod(a / v, 3)) for v in [81, 27, 9, 3, 1]]
        c = np.concatenate(ps)[:np.prod(shape)].reshape(shape)
        return (c - 1).astype(np.int8)

    # zero-run encoding
    def zr_encode(self, a):
        places = (a == 121).astype(np.int8)
        runs = [sum(1 for i in g) for k, g in groupby(places) if k]

        if len(runs) == 0:
            return a

        diff = places - np.roll(places, 1)
        diff[0] = places[0]
        starts = np.nonzero(diff == 1)[0]

        out_len = len(a)
        for r in runs:
            out_len -= (13 * (r // 14))
            out_len -= ((r % 14) - 1)
        out = np.zeros([out_len], dtype=np.uint8)

        in_idx_0 = 0
        in_idx_1 = starts[0]
        out_idx_0 = 0
        out_idx_1 = starts[0]
        for i in range(len(runs)):
            out[out_idx_0:out_idx_1] = a[in_idx_0:in_idx_1]
            n_14s = runs[i] // 14
            rem = runs[i] % 14
            in_idx_0 += (n_14s * 14) + rem

            for j in range(n_14s):
                out[out_idx_1 + j] = 255
            if rem > 1:
                out[out_idx_1 + n_14s] = 243 + (rem - 2)
            elif rem > 0:
                out[out_idx_1 + n_14s] = 121

            out_idx_1 += (n_14s + 1) if rem > 0 else n_14s
            out_idx_0 = out_idx_1
            in_idx_0 = starts[i] + runs[i]
            in_idx_1 = starts[i + 1] if i < len(runs) - 1 else len(a)
            out_idx_1 += in_idx_1 - in_idx_0
        out[out_idx_0:out_idx_1] = a[in_idx_0:in_idx_1]

        return out

    # zero-run decoding
    def zr_decode(self, z):
        gt242 = np.where(z > 242)[0]

        if len(gt242) == 0:
            return z

        extra_vals = np.sum(z[gt242]) - (242 * len(gt242))
        out = np.zeros([int(len(z) + extra_vals)], dtype=np.uint8)

        idx0 = 0
        idx1 = gt242[0]
        zidx0 = 0
        zidx1 = gt242[0]
        for i in range(len(gt242)):
            out[idx0:idx1] = z[zidx0:zidx1]
            run_len = z[gt242[i]] - 241
            out[idx1:idx1+run_len] = 121
            idx0 = idx1 + run_len
            zidx0 = zidx1 + 1
            zidx1 = gt242[i+1] if i < len(gt242) - 1 else len(z)
            idx1 = idx0 + (zidx1 - zidx0)
        out[idx0:idx1] = z[zidx0:zidx1]

        return out

    def recv_3LC_array_from_socket(self, conn):
        header = self.traff_comp.read_header(conn)
        if "STOP" in header:
            self.traff_comp.send_ack(conn)
            return "STOP", None, None

        vals = header.split(',')
        zr_data_len = int(vals[0])
        zr_len = int(vals[1])
        m = np.float32(vals[2])
        orig_shape = [int(n) for n in vals[3:]]
        zr_data = np.empty([zr_len], dtype=np.uint8)
        self.traff_comp.send_ack(conn)
        p = conn.recv_into(zr_data, zr_data_len, socket.MSG_WAITALL)
        self.traff_comp.send_ack(conn)

        return zr_data, orig_shape, m

    def send_3LC_array_over_socket(self, conn, zr_array, orig_shape, m):
        zr_data = zr_array.tobytes()
        zr_len = zr_array.shape[0]
        header = str(len(zr_data)) + ','
        header += str(zr_len) + ','
        header += str(m) + ','
        for s in orig_shape:
            header += str(s) + ','
        header = header[:-1] + '|'
        header = bytes(header, 'utf8')
        conn.send(header)
        self.traff_comp.check_resp(conn)
        conn.send(zr_data)
        self.traff_comp.check_resp(conn)

class TCPServer:
    def __init__(self, ipaddress, port, listen_num):
        self.modelAgg = ModelAggregate()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ipaddress, port))
        self.server_socket.listen(listen_num)

        self.inputs = [self.server_socket]
        self.outputs = []
        self.send_dict = {}
        self.count_dict = {}

        print('server.socket', self.server_socket)

        self.client_name_list = []
        self.name_list = [] # help to count the number of client finish to sent data
        self.rec_client_list = [] # collect the rec list from each client
        self.arr_agg = None
        self.client_sockets = []
        self.round_finish = False
        self.connected_client_num = 0
        self.count_roundNumber = 1 # include init round

        self.temp = [] #???

        self.data_comp = DataCompression()

    def accept_client(self):
        connection, client_address = self.server_socket.accept()
        print("Connection from {}".format(client_address))
        self.connected_client_num += 1
        self.inputs.append(connection)
        self.client_sockets.append(connection)
        self.send_dict[connection] = {"flag":protocol.SEND_INIT,"mess_data": self.modelAgg.getNetwork()}
        self.init_handler(connection)
        self.count_dict[connection] = 0

        print("count number ", self.connected_client_num)
        if self.connected_client_num == client_num:
            print("begin to send data!")
            for socket in self.client_sockets:
                self.outputs.append(socket)

    def _recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def tcp_read(self, sock):
        lengthbuf = self._recvall(sock, 4)
        if lengthbuf == None:
            return lengthbuf
        length, = struct.unpack('!I', lengthbuf)
        data = self._recvall(sock, length)
        message = pickle.loads(data)
        return message

    def tcp_send(self, sock, message):
        data = pickle.dumps(message)
        length = len(data)
        sock.sendall(struct.pack('!I', length))
        sock.sendall(data)
        if sock in self.outputs:
            self.outputs.remove(sock)

    def encode_data(self, data):

        encode_dict_array = []
        for i in range(len(data)):
            encode_dict = {}
            q, m = self.data_comp.quantize(data[i],compress_level)
            a = self.data_comp.q_encode(q)
            z = self.data_comp.zr_encode(a)
            encode_dict['z'] = z
            encode_dict['t_shape'] = data[i].shape
            encode_dict['m'] = m
            encode_dict_array.append(encode_dict)

        var_dict_dump = pickle.dumps(encode_dict_array)

        return var_dict_dump

    def decode_data(self,data): #zr,sr,mr

        var_array = pickle.loads(data)
        decode_array = []

        for i in var_array:
            zr = i['z']
            d = self.data_comp.zr_decode(zr)
            sr = i['t_shape']
            b = self.data_comp.q_decode(d, sr)
            mr = i['m']
            dq = self.data_comp.dequantize(b, mr)
            decode_array.append(np.array(dq))

        return decode_array

    def init_handler(self, socket):
        self.send_dict[socket]["flag"] = protocol.SEND_INIT
        weight = self.modelAgg.getNetwork()

        #encode_data = self.encode_data(weight)
        #self.send_dict[socket]["mess_data"] = encode_data

        self.send_dict[socket]["mess_data"] = weight

        print("SEND INIT DATA TO CLIENT")

    def para_agg_handler(self,socket,data):
        print("Rece para from client")

        decode_array = self.decode_data(data)

        if self.arr_agg == None:
            self.arr_agg = decode_array
        else:
            for i, arr in enumerate(decode_array):
                self.arr_agg[i] += arr

    def cal_para_handler(self):
        #self.send_dict[socket]["flag"] = protocol.SEND_PARA
        for i, arr in enumerate(self.arr_agg):
            self.arr_agg[i] = arr/client_num

        print('Cal para:',len(self.arr_agg))
        return self.arr_agg

    def round_handler(self,socket):
        arr_agg = self.cal_para_handler()

        #encode_data = self.encode_data(self.arr_agg)
        self.send_dict[socket]["flag"] = protocol.SEND_PARA
        #self.send_dict[socket]["mess_data"] = encode_data
        self.send_dict[socket]["mess_data"] = arr_agg
        print("SEND a round PARA TO CLIENT")

    def send_listback_handler(self, socket,data):
        self.send_dict[socket]["flag"]=protocol.SBS_TO_CLIENT_SEND_FINAL
        self.send_dict[socket]["mess_data"] = self.modelAgg.sbs_rec_list
        print("send list back to client")
        np.savetxt('result_p2p/' + str(config.dataset_name) + '_cachelist_' + str(top_num) + '.txt', self.modelAgg.sbs_rec_list)

    def end_handler(self,socket,data):
        print(" Round Finished.....   closing")
        print("   closing")
        if socket in self.outputs:
            self.outputs.remove(socket)
        self.inputs.remove(socket)
        socket.close()

    def run(self, timeout = 30):
        while self.inputs:
            readable, writable, exceptional = select.select(self.inputs, self.outputs, self.inputs, timeout)

            if not (readable or writable or exceptional):
                print("Time out ! ")
                break

            for s in readable:
                if s is self.server_socket:
                    self.accept_client()

                else:
                    data = self.tcp_read(s)
                    if data:
                        if data["flag"] == protocol.SEND_PARA:
                            self.count_dict[s] += 1
                            # print("self.count_dict[s]", self.count_dict[s], 'Round', self.round)
                            print('############# Client:', s.getpeername(), '############ Round:', self.count_dict[s]+1)
                            # count client number, then agg para
                            client_name = s

                            if client_name not in self.client_name_list:
                                self.client_name_list.append(client_name)
                                self.para_agg_handler(s, data["mess_data"])
                                print('Client list:', self.client_name_list, 'length:', len(self.client_name_list))

                            if len(self.client_name_list) == client_num:
                                for soc in self.client_name_list:
                                    if soc not in self.outputs:
                                        self.outputs.append(soc)
                                    self.round_handler(soc)
                                self.client_name_list = []
                                self.arr_agg = None
                                self.count_roundNumber += 1
                                print('A round finished',self.count_roundNumber)


                        if data["flag"] == protocol.CLIENT_LIST_TO_SBS:
                            #print("sum value is {} !!!!!!".format(sum(self.count_dict.values())))
                            soc_name=s
                            if soc_name not in self.name_list:
                                self.name_list.append(soc_name)
                                self.rec_client_list.append(data["mess_data"])

                            if len(self.name_list) == client_num:
                                self.modelAgg.countAcc(self.rec_client_list)
                                for soc_name in self.name_list:
                                    if soc_name not in self.outputs:
                                        self.outputs.append(soc_name)
                                    self.send_listback_handler(soc_name, data)
                                self.name_list = []
                                self.rec_client_list = []

                        if data["flag"] == protocol.Round_Finish:
                            if s not in self.temp:
                                self.temp.append(s)
                            if round_num == len(self.temp):
                                for soc in self.temp:
                                    self.end_handler(soc,data)

                    else:
                        # Interpret empty result as closed connection
                        print("   closing")
                        if s in self.outputs:
                            self.outputs.remove(s)
                        self.inputs.remove(s)
                        s.close()

            for s in writable:
                self.tcp_send(s, self.send_dict[s])

            for s in exceptional:
                print (" exception condition on ", s.getpeername())
                # stop listening for input on the connection
                self.inputs.remove(s)
                if s in self.outputs:
                    self.outputs.remove(s)
                s.close()