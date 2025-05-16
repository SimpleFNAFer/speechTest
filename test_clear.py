from functions import noise_reduction, text_recognition
import os

INPUT_DATA_DIR = os.path.join(os.curdir, 'wav', 'input_data')
INPUT_DATA_CLEAR_DIR = os.path.join(INPUT_DATA_DIR, 'clear', 'clear_test')
INPUT_DATA_NOISE_REDUCE_DIR = os.path.join(INPUT_DATA_DIR, 'noise_reduce', 'clear', 'clear_test')

paths = []

for f in os.scandir(INPUT_DATA_CLEAR_DIR):
    if f.is_file():
        paths.append(f.path)

print(paths)

for path in paths:
    print(path)
    noise_reduction(path)

paths = []

for f in os.scandir(INPUT_DATA_NOISE_REDUCE_DIR):
    if f.is_file():
        paths.append(f.path)

for path in paths:
    print(path)
    text_recognition(path)
