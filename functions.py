import os.path

from pyannote.audio import Pipeline
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
import json
from librosa import resample

from vosk import Model, KaldiRecognizer
import noisereduce as nr
import numpy as np
import soundfile as sf
import files


def diarization(path, token: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token)

    waveform, sample_rate = torchaudio.load(path)

    with ProgressHook() as hook:
        drz = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            hook=hook,
        )

    file_path = files.make_output_path(out_type=files.TYPE_DIAR, input_path=path)

    with open(file_path, "w") as rttm:
        drz.write_rttm(rttm)


def text_recognition(path: str):
    wf, rate = sf.read(path, dtype='int16')

    print(f"Audio length: {len(wf) / rate:.2f} seconds")
    print(f"Sample rate: {rate} Hz")

    if len(wf.shape) > 1:
        print('stereo')
        wf = wf.mean(axis=1)
    if rate != 16000:
        print('resampled')
        float_audio = wf.astype(np.float32) / 32767.0

        # Resample
        resampled_float = resample(float_audio, orig_sr=rate, target_sr=16000)

        # Scale back to int16 and clip
        wf = np.clip(resampled_float * 32767.0, -32768, 32767).astype(np.int16)
        rate = 16000

    model = Model(
        model_path=files.get_vosk_path(model_name='vosk-model-ru-0.42'),
        model_name="vosk-model-ru-0.42"
    )
    rec = KaldiRecognizer(model, rate)
    rec.SetWords(True)
    rec.SetPartialWords(True)

    text = []
    chunk_size = 4000
    for i in range(0, len(wf), chunk_size):
        chunk = wf[i:i + chunk_size].tobytes()
        if len(chunk) == 0:
            break
        if rec.AcceptWaveform(chunk):
            text.append(json.loads(rec.Result())["text"])

    text.append(json.loads(rec.FinalResult())["text"])

    file_path = files.make_output_path(out_type=files.TYPE_TR, input_path=path)

    f = open(file_path, "w", encoding='utf-8')
    f.write(" ".join(text))
    f.close()


def noise_reduction(path: str):
    data, rate = sf.read(path)

    if len(data.shape) > 1:
        print('stereo')
        data = data.mean(axis=1)

    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    file_path = files.make_output_path(out_type=files.TYPE_NR, input_path=path)
    sf.write(file_path, reduced_noise, rate)
