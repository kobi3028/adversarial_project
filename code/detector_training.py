import numpy as np
import os
from Detector_PCA_MLP import *
import time

VERBOSE = True
EXT = '.npy'

LAYER = 21
PARTS = 17
SAMPLES = 8313

def main(data_dir_name, plot_roc_graph, roc_graph_file_name):
    start = time.time()
    m_detector = Detector_PCA_MLP(LAYER, SAMPLES)

    # input handling

    for i in range(LAYER):
        print('[{:11.2f}s][+] train activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.partial_fit_activation_spaces(i, f_x_train)

    print('[{:11.2f}s][+] train layers activation space: Done'.format(time.time()-start))

    for i in range(LAYER):
        count = 0
        print('[{:11.2f}s][+] compute benign activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.compute_benign(i, count, f_x_train)
            count += len(f_x_train)

    print('[{:11.2f}s][+] compute benign activation space: Done'.format(time.time()-start))

    for i in range(LAYER):
        count = 0
        print('[{:11.2f}s][+] compute adversarial activation space: layer {}/21'.format(time.time() - start, i + 1))
        for j in range(PARTS):
            x_adv = np.load(os.path.join(data_dir_name, 'adv', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.compute_adversarial(i, count, x_adv)
            count += len(x_adv)

    print('[{:11.2f}s][+] compute adversarial activation space: Done'.format(time.time()-start))

    graph_dir_path = os.path.join(data_dir_name, 'roc_graphs')
    if not os.path.isdir(graph_dir_path):
        os.mkdir(graph_dir_path)

    if os.path.exists(os.path.join(graph_dir_path, roc_graph_file_name)):
        tmp = roc_graph_file_name.split('.')
        roc_graph_file_name = '{}_{}.{}'.format(tmp[0], len(os.listdir(graph_dir_path))+1, tmp[1])


    x_labels = np.load(os.path.join(data_dir_name, 'labels_benign' + EXT))
    m_detector.finish_fit(x_labels, plot_roc_graph=plot_roc_graph, roc_graph_file_name=os.path.join(graph_dir_path, roc_graph_file_name))

    m_detector.dump(data_dir_name, Detector.DEFAULT_FILE_NAME)
    # m_detector.load(data_dir_name, Detector.DEFAULT_FILE_NAME)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_training.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name, True, 'roc_graph.png')


