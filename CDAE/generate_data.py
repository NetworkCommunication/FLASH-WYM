import numpy

from keras.utils.np_utils import to_categorical


def load_data_100k():
    max_item_id  = -1
    train_history = {}
    with open('data/ml-100k/ua.base', 'r') as file:
        for line in file:
            user_id, item_id, rating, timestamp = line.rstrip().split('\t')
            if int(user_id) not in train_history:
                train_history[int(user_id)] = [int(item_id)]
            else:
                train_history[int(user_id)].append(int(item_id))

            if max_item_id < int(item_id):
                max_item_id = int(item_id)

    test_history = {}
    with open('data/ml-100k/ua.test', 'r') as file:
        for line in file:
            user_id, item_id, rating, timestamp = line.rstrip().split('\t')
            if int(user_id) not in test_history:
                test_history[int(user_id)] = [int(item_id)]
            else:
                test_history[int(user_id)].append(int(item_id))

    max_item_id += 1 # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)
        train_x[i] = numpy.sum(mat, axis=0)

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x

def load_data_1m():
    train_x = numpy.loadtxt("data/input_matrix_binary.txt", dtype=int)
    test_x = numpy.loadtxt("data/label_matrix_binary.txt", dtype=int)
    train_users = []
    test_users = []
    for i in range(train_x.shape[0]):
        train_users.append(i+1)
        test_users.append(i+1)
    return train_users, train_x, test_users, test_x

def load_data_1m_2():
    train_x = numpy.loadtxt("data/train_user1.txt")
    test_x = numpy.loadtxt("data/test_user1.txt")
    train_x[train_x>0] = 1
    test_x[test_x>0] = 1
    train_x = numpy.array(train_x, dtype=int)
    test_x = numpy.array(test_x, dtype=int)
    print(test_x)
    train_users = []
    test_users = []
    for i in range(train_x.shape[0]):
        train_users.append(i+1)
        test_users.append(i+1)
    return train_users, train_x, test_users, test_x