from pyannote.audio import Pipeline
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from asteroid.models import ConvTasNet
import wave
import json
from vosk import Model, KaldiRecognizer
import noisereduce as nr
from scipy.io import wavfile


def perform_diarization(path, token: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token)

    waveform, sample_rate = torchaudio.load(path)

    with ProgressHook() as hook:
        diarization = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            hook=hook,
        )

    # dump the diarization output to disk using RTTM format
    with open(f"{path}_DIAR.rttm", "w") as rttm:
        diarization.write_rttm(rttm)


def perform_separation(path: str):
    # 'from_pretrained' automatically uses the right model class (asteroid.models.DPRNNTasNet).
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    model.separate(path, resample=True)


def perform_text_recognition(path: str):
    wf = wave.open(path, "rb")

    model = Model(model_name="vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        # if silence detected save result
        if rec.AcceptWaveform(data):
            text.append(json.loads(rec.Result())["text"])

    text.append(json.loads(rec.FinalResult())["text"])

    f = open(f"{path}_sp_recognize.txt", "w")
    f.write(" ".join(text))
    f.close()

def perform_noise_reduction(path: str):
    rate, data = wavfile.read(path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(f"{path}_noise_red.wav", rate, reduced_noise)
