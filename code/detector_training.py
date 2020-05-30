import numpy as np
from detector import *
import time

VERBOSE = True
EXT = '.npy'


def main(data_dir_name, plot_roc_graph, roc_graph_file_name):
    start = time.time()
    for i in range(21):
        #input handling
        m_detector = Detector()

        for j in range(17):
            print('[{:11.2f}s][+] train activation space: part {}/17'.format(time.time()-start, j+1))
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.partial_fit_activation_spaces(f_x_train)

        print('[{:11.2f}s][+] train activation space: Done'.format(time.time()-start))

        for j in range(17):
            print('[{:11.2f}s][+] compute benign activation space: part {}/17'.format(time.time()-start, j+1))
            f_x_train = np.load(os.path.join(data_dir_name, 'benign', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.compute_benign(f_x_train)

        print('[{:11.2f}s][+] compute benign activation space: Done'.format(time.time()-start))

        for j in range(17):
            print('[{:11.2f}s][+] compute adversarial activation space: part {}/17'.format(time.time()-start, j + 1))
            x_adv = np.load(os.path.join(data_dir_name, 'adv', 'layer_{}_{}'.format(i, j) + EXT))
            m_detector.compute_adversarial(x_adv)

        print('[{:11.2f}s][+] compute adversarial activation space: Done'.format(time.time()-start))

        x_labels = np.load(os.path.join(data_dir_name, 'y_benign' + EXT))
        m_detector.finish_fit(x_labels, plot_roc_graph=plot_roc_graph, roc_graph_file_name=roc_graph_file_name+str(i))

        m_detector.dump(data_dir_name, Detector.DEFAULT_FILE_NAME+str(i))
        # m_detector.load(data_dir_name, Detector.DEFAULT_FILE_NAME)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_training.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name, False, 'roc_graph.png')


