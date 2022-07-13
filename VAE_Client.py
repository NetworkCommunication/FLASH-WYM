import socket
import pickle
import struct
import select
import config
import protocol
import numpy as np
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from VAE import ANNVAE
from VAE import vae_loss
from CDAE import mask
import matplotlib.pyplot as plt
from numpy.random import random, randint
from itertools import groupby
from threading import Thread
from utils import *
import progressbar as pb
from collections import Counter
import io
import time
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config_gpu = tf.ConfigProto()
#config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.2
#set_session(tf.Session(config=config_gpu))

client_num = 1

top_num = config.top_num
sbs_num = config.sbs_num
round_num = config.round_num
compress_level = config.compress_level
client_index_list = config.client_index_list

class FLClientModel:
    def __init__(self, index, datasize,dataset_name):
        np.random.seed(1337)  # for reproducibility
        self.start_time= time.time()
        self.c_index=index
        if dataset_name == 'MovieLens':
            self.loadData(index, datasize)
        elif dataset_name == 'Netflix':
            self.loadNetflixData(index, datasize)
        self.initNetwork()
        self.recall_list_sbs = [] #record the recall accuarcy and help to count the round number

    def loadNetflixData(self,client_index, datasize):
        print('Start to load data')

        RatingMatrix = np.loadtxt('data/Netflix_Data/training.txt')
        TestMatrix = np.loadtxt('data/Netflix_Data/testing.txt')
        RatingMatrix = RatingMatrix[:,0:7000]
        TestMatrix = TestMatrix[:,0:7000]

        print('data start:', datasize * (client_index - 1))
        print('data end:', datasize * client_index)

        self.RatingMatrix = RatingMatrix[datasize * (client_index - 1):datasize * client_index, :]
        self.TestMatrix = TestMatrix[datasize * (client_index - 1):datasize * client_index, :]

        print('RatingMatrix Shape:', self.RatingMatrix.shape)

        self.RatingMatrix[self.RatingMatrix > 0] = 1
        self.TestMatrix[self.TestMatrix > 0] = 1

        self.RatingMatrix = self.RatingMatrix.astype(np.float32)
        self.TestMatrix = self.TestMatrix.astype(np.float32)

        self.total_train_num = np.sum(self.RatingMatrix)

        sum = 0
        for i in client_index_list:
            RatingMatrix_new = RatingMatrix[datasize * (i - 1):datasize * i]
            print(RatingMatrix_new.shape)
            RatingMatrix_new[RatingMatrix_new > 0] = 1
            num = np.sum(RatingMatrix_new)
            sum = sum + num

        print("~~~~~~~~~~~~~~~~~~~SUM~~~~~~~~~~~~~~", sum)
        self.weight_tr_num = self.total_train_num / sum
        print("~~~~~~~~~~~~~~~~~~~Weight SUM ~~~~~~~~~~~~~~", self.weight_tr_num)

    def loadData(self, client_index, datasize):
        print('Start to load data')
        RatingMatrix = np.loadtxt('data/MovieLens_Data/training_by_age_all.txt')
        #RatingMatrix = np.loadtxt('data/MovieLens_Data/train_matrix_zipcode_sorted.txt')
        #RatingMatrix = np.array(RatingMatrix)[:, 0:3883]
        #self.RatingMatrix = np.loadtxt('data/MovieLens_Data/train.txt')

        TestMatrix = np.loadtxt('data/MovieLens_Data/testing_by_age_all.txt')
        #TestMatrix = np.loadtxt('data/MovieLens_Data/test_matrix_zipcode_sorted.txt')
        #TestMatrix = np.array(TestMatrix)[:, 0:3883]
        #self.TestMatrix = np.loadtxt('data/MovieLens_Data/test.txt')

        print('data start:', datasize * (client_index - 1))
        print('data end:', datasize * client_index)

        self.RatingMatrix = RatingMatrix[datasize * (client_index - 1):datasize * client_index, :]
        self.TestMatrix = TestMatrix[datasize * (client_index - 1):datasize * client_index, :]
        # self.UserAttr = UserAttr[datasize * (client_index - 1):datasize * client_index, :]

        print('RatingMatrix Shape:', self.RatingMatrix.shape)

        self.RatingMatrix[self.RatingMatrix > 0] = 1
        self.TestMatrix[self.TestMatrix > 0] = 1

        self.RatingMatrix = self.RatingMatrix.astype(np.float32)
        self.TestMatrix = self.TestMatrix.astype(np.float32)

        self.total_train_num = np.sum(self.RatingMatrix)

        sum = 0
        for i in client_index_list:
            RatingMatrix_new = RatingMatrix[datasize * (i - 1):datasize * i]
            print(RatingMatrix_new.shape)
            RatingMatrix_new[RatingMatrix_new > 0] = 1
            num = np.sum(RatingMatrix_new)
            sum = sum + num

        print("~~~~~~~~~~~~~~~~~~~SUM~~~~~~~~~~~~~~", sum)
        self.weight_tr_num = self.total_train_num / sum
        print("~~~~~~~~~~~~~~~~~~~Weight SUM ~~~~~~~~~~~~~~", self.weight_tr_num)

    def get_mask(self, data):
        return mask.get_mask(data, "user")

    def initNetwork(self):
        # deep autoencoders based on Keras
        self.vae_model = ANNVAE()
        #self.vae_model.variables

    def setNetwork(self, weight):
        self.vae_model.set_weights(weight)

    def getNetwork(self):
        weight = self.vae_model.get_weights()
        #print('############weight##########', len(weight))
        #print('############weight##########',weight)
        return weight

    def train(self, epochs):
        # vae_optimizer = tf.train.AdamOptimizer(1e-10) #learning rate 1e-10

        vae_optimizer = tf.train.GradientDescentOptimizer(config.learning_rate) #learning rate 1e-4

        train_dataset = tf.data.Dataset.from_tensor_slices(self.RatingMatrix).shuffle(config.shuffle_size).batch(config.batch_size)

        for epoch in range(epochs):
            vae_loss_value_batch = []

            for data in train_dataset:
                with tf.GradientTape() as vae_tape:
                    x_hat, z_mu, z_log_sigma_sq = self.vae_model(data, training=True)

                    vae_loss_value = vae_loss(data, x_hat, z_mu, z_log_sigma_sq)
                    vae_loss_value_batch.append(vae_loss_value)
                #print("variable: ", self.vae_model.get_weights())
                gradients_vae = vae_tape.gradient(vae_loss_value, self.vae_model.variables)
                gradients_vae, norm_gradients = tf.clip_by_global_norm(gradients_vae, config.clip_val)

                # gradients_vae_list = []
                # for grad in gradients_vae:
                #     grad_value = tf.clip_by_value(
                #         grad,
                #         clip_value_min=-0.000001,
                #         clip_value_max= 0.000001
                #     )
                #     gradients_vae_list.append(grad_value)
                #
                # gradients_vae = gradients_vae_list

                # print("gradients: ", gradients_vae)

                vae_optimizer.apply_gradients(zip(gradients_vae, self.vae_model.variables))

            #if epoch % 1 == 0:
                pred, _, _ = self.vae_model(self.RatingMatrix, training=False)
                #print("loss:", np.mean(vae_loss_value_batch))
                #print( np.array(self.RatingMatrix[0:batch_size] == 0).shape)
                #print('pred:',pred)

        return pred

    def predict(self,pred):
        pred = pred * (self.RatingMatrix == 0)  # remove watched items from predictions
        pred = np.argsort(pred)
        rec = pred[:, -top_num:]
        recommended = np.zeros(self.RatingMatrix.shape)
        for i in range(rec.shape[0]):
            recommended[i][rec[i]] = 1
        rec_sum = recommended.sum(axis=0)
        rec_sum_sort = np.argsort(rec_sum)
        cache = rec_sum_sort[-top_num:]
        rec_list = cache.tolist()
        np.savetxt('result_p2p/' + str(config.dataset_name) + '_cachelist_' + str(top_num) + '_SBS_User' + str(self.c_index) + '.txt',
                   rec_list)
        return rec_list

        #self.test(rec_list)
        #self.validation_data(rec_list)
        #print(self.rec_list)

    def test_data(self,serverList):

        print('TO BE TEST LIST:', len(serverList))
        self.sbs_list = serverList[0:top_num]

    def validation_data(self,rec_list):
        train_popular_num = self.RatingMatrix.sum(axis=0, dtype='float')

        top_train_popular_num = (-train_popular_num).argsort()[0:top_num]

        test_popular_num = self.TestMatrix.sum(axis=0, dtype='float')
        top_test_popular_num = (-test_popular_num).argsort()[0:top_num]

        count_train_val = 0
        count_test_val = 0
        count_all = 0
        same_list = []

        cache_list = []
        for id in rec_list:
            if id not in cache_list:
                cache_list.append(id)

        for i in cache_list:
            for j in top_train_popular_num:
                if int(i) == int(j):
                    count_train_val += 1
            for q in top_test_popular_num:
                if int(i) == int(q):
                    count_test_val += 1

        for tl in top_train_popular_num:
            for ttl in top_test_popular_num:
                if tl == ttl:
                    if tl in same_list:
                        continue
                    else:
                        count_all += 1
                        same_list.append(ttl)

        print("The same number with train_data is", count_train_val, count_train_val / top_num)
        print("The same number with test_data is", count_test_val, count_test_val / top_num)
        print("The same number between train and test is", count_all)

    def test(self, cacheList,flag_bs):

        cache_list = []
        for id in cacheList:
            if id not in cache_list:
                cache_list.append(id)

        count = self.TestMatrix.shape[0]
        rcl = 0

        for i in range(0, self.TestMatrix.shape[0]):  # self.rec_top_N(2)
            rating = self.TestMatrix[i, :]
            vindex = np.where(rating > 0)
            vindex = np.array(vindex)
            if vindex.shape[1] == 0:
                print("Ignored all zero ...")
                count -= 1

            rtrue = 0
            user_req = vindex[0, :]

            for rlist in cache_list:
                for ulist in user_req:
                    if rlist == ulist:
                        rtrue += 1
                        # tol_recall += 1

            if vindex.shape[1] > 0:
                # ML
                recall = rtrue / vindex.shape[1]
                rcl = rcl + recall

        if (rcl / count > 1):
            np.savetxt("data/wrong.txt", cacheList)

        if flag_bs == protocol.TEST_SBS:
            self.recall_list_sbs.append(rcl / count)
            print("The cache efficiency@", top_num, " is: ", rcl / count)
            print("The recall SBS list:", self.recall_list_sbs)


class LossyCompression:
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
        n_0s = int(np.ceil(len(t_flat) / 5.0) * 5.0) - len(t_flat)
        p = np.split(np.pad(t_flat, (0, n_0s), 'constant'), 5)
        a = (p[0] * 81) + (p[1] * 27) + (p[2] * 9) + (p[3] * 3) + p[4]
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
            out[idx1:idx1 + run_len] = 121
            idx0 = idx1 + run_len
            zidx0 = zidx1 + 1
            zidx1 = gt242[i + 1] if i < len(gt242) - 1 else len(z)
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


class TCPClient:
    def __init__(self,ipaddress, port,index,datasize,dataset_name):
        self.round_num = round_num
        self.lo_cm = LossyCompression()
        address = (ipaddress, port)
        self.outputs = []
        self.inputs = []
        self.socks = []
        self.outputs = []
        self.c_index = index
        self.send_dict = {}
        self.round_dict ={}
        self.client_model = FLClientModel(index,datasize,dataset_name)

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(address)
        self.inputs.append(client_socket)
        self.send_dict[client_socket] = {"flag": protocol.SEND_INIT, "mess_data": []}
        self.round_dict[client_socket] = 0
        #count+=1

        self.data_comp_dict_save=[] #for save txt

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
        length, = struct.unpack('!I', lengthbuf)
        data = self._recvall(sock, length)
        message = pickle.loads(data)
        self.outputs.append(sock)

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
            data_comp_dict = {} # for save txt
            q, m = self.lo_cm.quantize(data[i],compress_level)
            a = self.lo_cm.q_encode(q)
            z = self.lo_cm.zr_encode(a)
            encode_dict['z'] = z
            encode_dict['t_shape'] = data[i].shape
            encode_dict['m'] = m
            encode_dict_array.append(encode_dict)
            #save to txt
            print('#', i, '## data new shape:', data[i].shape, 'a:', a.shape, 'z:', z.shape, 'm:', m)
            data_comp_dict['data_shape'] = data[i].shape
            data_comp_dict['a'] = a.shape
            data_comp_dict['z'] = z.shape
            data_comp_dict['m'] = m
            self.data_comp_dict_save.append(data_comp_dict)

        var_dict_dump = pickle.dumps(encode_dict_array)

        f = open('result/DataComp/'+ str(config.dataset_name) + '_' + str(top_num) + '_DataComp'+str(config.compress_level)+'_'+ str(self.c_index) + '.txt', 'w')
        f.write(str(self.data_comp_dict_save))
        f.close()

        return var_dict_dump

    def decode_data(self,data): #zr,sr,mr

        var_array = pickle.loads(data)
        decode_array = []

        for i in var_array:
            zr = i['z']
            d = self.lo_cm.zr_decode(zr)
            sr = i['t_shape']
            b = self.lo_cm.q_decode(d, sr)
            mr = i['m']
            dq = self.lo_cm.dequantize(b, mr)
            decode_array.append(np.array(dq))

        return decode_array

    def para_list_handler(self,socket,data):
        # decode data
        #de_data = self.decode_data(data)
        #set weight
        self.client_model.setNetwork(data)
        pred_rec = self.client_model.train(config.epochs)
        rec_list = self.client_model.predict(pred_rec)

        self.send_dict[socket]["flag"] = protocol.CLIENT_LIST_TO_SBS
        self.send_dict[socket]["mess_data"] = rec_list

    def para_handler(self,socket):

        WB = self.client_model.getNetwork()
        weight_WB = self.weight_para_handler(WB)

        #encode weighted WB
        var_dict_dump = self.encode_data(weight_WB)

        self.send_dict[socket]["flag"] = protocol.SEND_PARA
        self.send_dict[socket]["mess_data"] = var_dict_dump
        print("Send para to server")

    def weight_para_handler(self, WB):
        weight_WB = []
        for wb_i in WB:
            wb = wb_i * self.client_model.weight_tr_num
            weight_WB.append(wb)
        return weight_WB

    def rec_list_to_sbs_handler(self,socket,data):
        self.send_dict[socket]["flag"] = protocol.CLIENT_LIST_TO_SBS

        de_data = self.decode_data(data)

        self.client_model.setNetwork(de_data)
        pred_rec = self.client_model.train(config.epochs)
        rec_list = self.client_model.predict(pred_rec)
        print('Predict cache list:', len(rec_list))
        self.send_dict[socket]["mess_data"] = rec_list

    def test_handler(self,socket,data):
        self. client_model.test_data(data)
        print('#################### TEST SBS ##########################')
        self.client_model.test(self.client_model.sbs_list,protocol.TEST_SBS)
        self.client_model.validation_data(self.client_model.sbs_list)

        if (len(self.client_model.recall_list_sbs)==self.round_num):
            self.send_dict[socket]["flag"] = protocol.Round_Finish
            self.send_dict[socket]["mess_data"] = ['ROUND END']

            self.save_txt(self.client_model.recall_list_sbs,self.client_model.sbs_list)
            #self.save_cache_eff_plot(self.client_model.recall_list_sbs)

            print("####### END!!!! ######")

            if socket in self.outputs:
                self.outputs.remove(socket)
            self.inputs.remove(socket)
        else:
            self.para_handler(socket)


    def save_txt(self,data,c_list):
        time_list = []
        time_list.append(self.client_model.start_time)
        self.finish_time = time.time()
        time_list.append(self.finish_time)
        # save result
        np.savetxt('result_p2p/'+ str(config.dataset_name) + '_' + str(top_num) + '_SBS_User' + str(self.c_index) + '.txt',data)
        # save sbs cache list
        # np.savetxt('result_new/' + str(config.dataset_name) + '_cachelist_' + str(top_num) + '_SBS_User' + str(self.c_index) + '.txt',
        #            c_list)
        np.savetxt('result_p2p/' + str(config.dataset_name) + '_time_' + str(top_num) + '_SBS_User' + str(self.c_index) + '.txt',
                   time_list)

    def save_cache_eff_plot(self,data):
        # plot
        x_space = 1
        x_lim = np.arange(0, round_num, x_space)
        x_value = [int(i) for i in x_lim]
        y_value = [j * 100 for j in data]
        plt.figure()
        plt.plot(x_value, y_value)
        #plt.xlim(0, round_num)
        plt.ylim(0, max(y_value)+10)
        plt.xlabel('Round Number')
        plt.ylabel('Cache Efficiency')
        plt.savefig('result/'+ str(config.dataset_name) + '_' + str(top_num) + '_SBS_User' + str(self.c_index) + '.pdf')

    def run(self, timeout=30):
        while self.inputs:
            #print("Waiting events!")
            readable, writable, exceptional = select.select(self.inputs, self.outputs, self.inputs, timeout)

            if not (readable or writable or exceptional):
                print("Time out ! ")
                break

            for s in readable:
                data = self.tcp_read(s)

                #print('Recevie FLAG:',data["flag"])
                if data["flag"] == protocol.SEND_INIT:
                    print(s.getpeername(), "received init data")
                    self.para_list_handler(s,data["mess_data"])

                elif data["flag"] == protocol.SEND_PARA:
                    print(s.getpeername(), "received para data")
                    self.para_list_handler(s,data["mess_data"])

                elif data["flag"] == protocol.SBS_TO_CLIENT_SEND_FINAL:
                    print(s.getpeername(), "received agg cache list")
                    self.test_handler(s, data["mess_data"])
                else:
                    # Interpret empty result as closed connection
                    print("-----closing-----")
                    if s in self.outputs:
                        self.outputs.remove(s)
                    self.inputs.remove(s)
                    # s.close()

            for s in writable:
                #print('send data:', self.send_dict[s])
                self.tcp_send(s, self.send_dict[s])
