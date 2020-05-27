import numpy as np
from detector import *

VERBOSE = True
EXT = '.npy'


def main(data_dir_name, plot_roc_graph, save_to_file):
    #input handling
    tmp_1 = 'y_adv_target'
    tmp_2 = 'y_adv_original'

    f_x_train = np.load(os.path.join(data_dir_name, 'logits_on_x_benign' + EXT))
    x_labels  = np.load(os.path.join(data_dir_name, 'y_benign' + EXT))

    # Construct adversarial examples (input files)
    x_adv = np.load(os.path.join(data_dir_name, 'logits_on_x_adv' + EXT))

    m_detector = Detector()
    m_detector.fit(f_x_train, x_adv, x_labels, plot_roc_graph=True)
    m_detector.dump(data_dir_name, Detector.DEFAULT_FILE_NAME)
    m_detector.load(data_dir_name, Detector.DEFAULT_FILE_NAME)
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_training.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name, False, True)


