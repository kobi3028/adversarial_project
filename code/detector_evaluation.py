from detector import *
import pickle
import os


def predict_arr(m_detector, f_x_eval):
    return m_detector.predict(f_x_eval)


def main(base_dir_name):
    #load detector
    with open(os.path.join(base_dir_name, Detector.FILE_NAME), 'rb') as f:
        m_detector = pickle.load(f)

    f_x_eval = np.load(os.path.join(data_dir_name, 'evaloation_data' + EXT))

    predict(m_detector, f_x_eval)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: detector_evaluation.py <DIR_NAME>')
        exit()
    dir_name = sys.argv[1]
    main(dir_name)