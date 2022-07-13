from SBS_Server import TCPServer
import tensorflow as tf
tf.enable_eager_execution()

tcp_server = TCPServer("127.0.0.1", 10369, 10)
tcp_server.run(timeout=10000)