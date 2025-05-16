from functions import (
    diarization,
    text_recognition,
    noise_reduction,
)
import os
import yaml
import files

token_env_key = 'HF_TOKEN'

token = os.getenv(token_env_key)

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

if token is None:
    token = cfg.get(token_env_key)

data = files.get_wav_data('records')

print(data)

for path in data:
    noise_reduction(path)
    diarization(path, token=token)
    text_recognition(path)

data_noise_reduce = files.get_wav_data('NR')

for path in data_noise_reduce:
    diarization(path, token=token)
    text_recognition(path)
