import numpy as np
import os
from Detector_PCA_MLP import *
import time
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


VERBOSE = True
EXT = '.npy'

LAYER = 21
PARTS = 17
SAMPLES = 8313
TRAIN_SAMPLES = int(SAMPLES * 0.9)
TEST_SAMPLES = SAMPLES - TRAIN_SAMPLES
ALL_SAMPLES = SAMPLES


def main(data_dir_name, roc_graph_file_name, precision_recall_graph_file_name):
    start = time.time()

    count = 0
    for j in range(PARTS):
        f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(0, j) + EXT))
        count += len(f_x_train)

    print('[{:11.2f}s][+] number of samples:{}'.format(time.time() - start, count))
    print('[{:11.2f}s][+] train samples:{}\ttest samples:{}'.format(time.time() - start, TRAIN_SAMPLES, TEST_SAMPLES))
    random.seed(1337)
    all_indexes = [i for i in range(ALL_SAMPLES)]
    random.shuffle(all_indexes)
    train_indexes = all_indexes[:TRAIN_SAMPLES]
    test_indexes = all_indexes[TRAIN_SAMPLES:]
    # input handling

    m_detector = Detector_PCA_MLP(LAYER, len(train_indexes))

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

    graph_dir_path = os.path.join(data_dir_name, 'roc_graphs')
    if not os.path.isdir(graph_dir_path):
        os.mkdir(graph_dir_path)

    if os.path.exists(os.path.join(graph_dir_path, roc_graph_file_name)):
        tmp = roc_graph_file_name.split('.')
        roc_graph_file_name = '{}_{}.{}'.format(tmp[0], len(os.listdir(graph_dir_path)) + 1, tmp[1])

    x_labels = np.load(os.path.join(data_dir_name, 'labels_benign' + EXT))
    x_labels = [x_labels[i] for i in range(len(x_labels)) if i in train_indexes]
    m_detector.finish_fit(x_labels)

    print('[{:11.2f}s][+] model training: Done'.format(time.time() - start))

    Y_score_benign = [abs(m_detector.predict(test_data_benign[key])) for key in test_data_benign]
    Y_score_adv = [abs(m_detector.predict(test_data_adversrial[key])) for key in test_data_adversrial]
    Y_score = Y_score_benign + Y_score_adv
    y_tmp = np.array(Y_score)
    np.save(os.path.join(data_dir_name, 'prediction.npy'), y_tmp)
    Y_true = ([0] * len(test_data_benign)) + ([1] * len(test_data_adversrial))

    plt.figure(figsize=(50, 50))
    plt.bar(np.arange(len(Y_score_benign)), Y_score_benign, color='#7f6d5f', label='benign', align='edge')
    plt.bar(np.arange(len(Y_score_adv)) + len(Y_score_benign), Y_score_adv, color='000000', label='adversarial', align='edge')
    plt.legend()
    plt.savefig(os.path.join(data_dir_name, 'prediction_score_histogram.png'))
    #plt.show(block=True)
    plt.clf()

    lr_precision, lr_recall, _ = precision_recall_curve(Y_true, Y_score, pos_label=1)
    precision_recall_auc = auc(lr_recall, lr_precision)

    lw = 2
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.plot(lr_recall, lr_precision, color='darkorange', lw=lw, label='precision recall curve (area = %0.3f)' % precision_recall_auc)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall graph')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(graph_dir_path, precision_recall_graph_file_name))
    #plt.show(block=True)
    plt.clf()

    fpr, tpr, thresholds = roc_curve(Y_true, Y_score, pos_label=1)
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

    plt.savefig(os.path.join(graph_dir_path, roc_graph_file_name))
    #plt.show(block=True)
    plt.clf()

    m_detector.dump(data_dir_name, Detector_PCA_MLP.DEFAULT_FILE_NAME)
    # m_detector.load(data_dir_name, Detector.DEFAULT_FILE_NAME)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_training.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name, 'roc_graph_test.png', 'precision_recall_graph_test.png')


