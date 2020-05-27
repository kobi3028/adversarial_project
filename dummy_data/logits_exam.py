import numpy as np
from pathlib import Path
import shutil
import os

#creates logits examples
number_of_examples = 20
number_of_layers = 18
layer_length = 256
layer_width = 256
lowest_y_value = 0
highest_y_value = 9

logits_on_x_benign = np.random.rand(number_of_examples, number_of_layers, layer_length, layer_width)
logits_on_x_adv = np.random.rand(number_of_examples, number_of_layers, layer_length, layer_width)
y_benign = np.random.randint(0, high=10, size=(number_of_examples))
y_adv_original = np.random.randint(lowest_y_value, high=(highest_y_value+1), size=(number_of_examples))
y_adv_target = np.random.randint(lowest_y_value, high=(highest_y_value+1), size=(number_of_examples))

dirpath = Path('data')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

os.mkdir('data')

np.save(r'./data/logits_on_x_benign', logits_on_x_benign)
np.save(r'./data/logits_on_x_adv',    logits_on_x_adv)
np.save(r'./data/y_benign',           y_benign)
np.save(r'./data/y_adv_original',     y_adv_original)
np.save(r'./data/y_adv_target',       y_adv_target)