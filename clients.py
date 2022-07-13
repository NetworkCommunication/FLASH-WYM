import tensorflow as tf
tf.enable_eager_execution()

from VAE_Client import TCPClient
import config

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('-i', '--index', default=1, type=int, help='index')
args = parser.parse_args()

tcp_client = TCPClient("127.0.0.1", 10369, args.index, config.Movie_lens_data_size, config.dataset_name)
tcp_client.run(timeout=10000)