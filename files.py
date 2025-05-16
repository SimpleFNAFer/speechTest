import os

WAV_DATA_DIR = os.path.join(os.curdir, 'wav')
TEXT_DATA_DIR = os.path.join(os.curdir, 'text')

TYPE_DIAR = 'DIAR'
TYPE_TR = 'TR'
TYPE_NR = 'NR'

OUTPUT_DATA_DIARIZATION_DIR = os.path.join(TEXT_DATA_DIR, 'diarization')
OUTPUT_DATA_TEXT_RECOGNITION_DIR = os.path.join(TEXT_DATA_DIR, 'text_recognition')


def get_vosk_path(model_name: str) -> str:
    path = os.path.join(os.curdir, 'vosk', model_name)
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f'{path} not found')


def get_wav_data(*search_paths) -> list[str]:
    paths = []

    search_path = os.path.join(WAV_DATA_DIR, *search_paths)

    for c_d, _, c_fs in os.walk(search_path):
        for c_f in c_fs:
            if c_f.lower().endswith('.wav'):
                full_path = os.path.join(c_d, c_f)
                paths.append(full_path)

    return paths


def make_output_path(out_type: str, input_path: str) -> str:
    head, tail = os.path.split(input_path)
    rel_path = os.path.relpath(head, WAV_DATA_DIR)
    file, ext = os.path.splitext(tail)

    if out_type == TYPE_DIAR:
        out_dir = TEXT_DATA_DIR
        out_ext = '.rttm'
    elif out_type == TYPE_TR:
        out_dir = TEXT_DATA_DIR
        out_ext = '.txt'
    elif out_type == TYPE_NR:
        out_dir = WAV_DATA_DIR
        out_ext = '.wav'
    else:
        raise TypeError('invalid output type')

    f = os.path.join(out_dir, out_type, rel_path, f'{file}{out_ext}')
    d, _ = os.path.split(f)

    if not os.path.exists(d):
        os.makedirs(d)

    return f
