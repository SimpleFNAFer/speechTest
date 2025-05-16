import files
import wave

print(files.get_input_data_segmentation())

for path in files.get_input_data_segmentation():
    print(path)
    wf = wave.open(path, "rb")
    print(wf.tell())
