import numpy as np
import os
from detector import *
import time
import random


VERBOSE = True
EXT = '.npy'

LAYER = 21
PARTS = 17
TRAIN_SAMPLES = int(8313 * 0.9)
TEST_SAMPLES = 8313 - TRAIN_SAMPLES
ALL_SAMPLES = 8313


def main(data_dir_name, roc_graph_file_name):

    start = time.time()
    print('[{:11.2f}s][+] train samples:{}\ttest samples:{}'.format(time.time() - start, TRAIN_SAMPLES, TEST_SAMPLES))
    random.seed(os.urandom(5))
    all_indexes = [i for i in range(ALL_SAMPLES)]
    random.shuffle(all_indexes)
    train_indexes = all_indexes[:TRAIN_SAMPLES]
    test_indexes = all_indexes[TRAIN_SAMPLES:]
    # input handling

    m_detector = Detector(LAYER, len(train_indexes))

    test_data_benign = {i:[[] for j in range(LAYER)] for i in test_indexes}
    test_data_adversrial = {i:[[] for j in range(LAYER)] for i in test_indexes}

    for i in range(LAYER):
        count = 0
        print('[{:11.2f}s][+] train activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            train_data = []
            for k in range(len(f_x_train)):
                if count + k in train_indexes:
                    train_data.append(f_x_train[k])
                else:
                    test_data_benign[count + k][i] = f_x_train[k]
            m_detector.partial_fit_activation_spaces(i, train_data)
            count += len(f_x_train)

    print('[{:11.2f}s][+] train layers activation space: Done'.format(time.time()-start))

    for i in range(LAYER):
        count = 0
        i_count = 0
        print('[{:11.2f}s][+] compute benign activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            train_data = [f_x_train[k] for k in range(len(f_x_train)) if i_count + k in train_indexes]
            m_detector.compute_benign(i, count, train_data)
            i_count += len(f_x_train)
            count += len(train_data)

    print('[{:11.2f}s][+] compute benign activation space: Done'.format(time.time()-start))

    for i in range(LAYER):
        count = 0
        i_count = 0
        print('[{:11.2f}s][+] compute adversarial activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            x_adv = np.load(os.path.join(data_dir_name, 'adv', 'layer_{}_{}'.format(i, j) + EXT))
            train_data = []
            for k in range(len(x_adv)):
                if i_count + k in train_indexes:
                    train_data.append(x_adv[k])
                else:
                    test_data_adversrial[i_count + k][i] = x_adv[k]
            m_detector.compute_adversarial(i, count, train_data)
            count += len(train_data)
            i_count += len(x_adv)

    print('[{:11.2f}s][+] compute adversarial activation space: Done'.format(time.time()-start))

    graph_dir_path = os.path.join(data_dir_name, 'roc_graphs')
    if not os.path.isdir(graph_dir_path):
        os.mkdir(graph_dir_path)

    if os.path.exists(os.path.join(graph_dir_path, roc_graph_file_name)):
        tmp = roc_graph_file_name.split('.')
        roc_graph_file_name = '{}_{}.{}'.format(tmp[0], len(os.listdir(graph_dir_path))+1, tmp[1])

    x_labels = np.load(os.path.join(data_dir_name, 'labels_benign' + EXT))
    x_labels = [x_labels[i] for i in range(len(x_labels)) if i in train_indexes]
    m_detector.finish_fit(x_labels, plot_roc_graph=True)

    print('[{:11.2f}s][+] model training: Done'.format(time.time() - start))

    graph_dir_path = os.path.join(data_dir_name, 'roc_graphs')
    if not os.path.isdir(graph_dir_path):
        os.mkdir(graph_dir_path)

    if os.path.exists(os.path.join(graph_dir_path, roc_graph_file_name)):
        tmp = roc_graph_file_name.split('.')
        roc_graph_file_name = '{}_{}.{}'.format(tmp[0], len(os.listdir(graph_dir_path)) + 1, tmp[1])

    Y_score = [m_detector.predict(test_data_benign[key]) for key in test_data_benign] + [m_detector.predict(test_data_adversrial[key]) for key in test_data_adversrial]
    print(Y_score)
    Y_true = ([0] * len(test_data_benign)) + ([1] * len(test_data_adversrial))

    fpr, tpr, thresholds = roc_curve(Y_true, Y_score, pos_label=0)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic graph')
    plt.legend(loc="lower right")

    plt.savefig(roc_graph_file_name)
    plt.show()

    m_detector.dump(data_dir_name, Detector.DEFAULT_FILE_NAME)
    # m_detector.load(data_dir_name, Detector.DEFAULT_FILE_NAME)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_training.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name, 'roc_graph.png')


