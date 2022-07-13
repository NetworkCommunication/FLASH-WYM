import tensorflow as tf
tf.enable_eager_execution()

from VAE_Client import TCPClient
import config
import numpy as np

s_index = 5 #5

dataset_name = config.dataset_name

if dataset_name == 'Netflix':
    s_data_size = config.Netfliex_lens_data_size

elif dataset_name == 'MovieLens':
    s_data_size = config.Movie_lens_data_size

tcp_client = TCPClient("127.0.0.1", 10369, s_index, s_data_size, dataset_name)
tcp_client.run(timeout=10000)