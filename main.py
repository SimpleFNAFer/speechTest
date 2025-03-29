from pipeline import perform_diarization, perform_separation, perform_text_recognition, perform_noise_reduction
import os
import re

directory = 'wav'
token_env_key = 'HF_TOKEN'

paths = []
regWav = re.compile(r'^.*\.wav$')
regSep = re.compile(r'^.*est\d+\.wav$')
regNoise = re.compile(r'^.*_noise\.wav$')
regNoiseRed = re.compile(r'^.*_noise_red\.wav$')
regNoiseRedSep = re.compile(r'^.*_noise_red_est\d+\.wav$')

for entry in os.scandir(directory):
    if entry.is_file() and regWav.match(entry.path):
        paths.append(entry.path)

for path in paths:
    perform_diarization(path, os.getenv(token_env_key))
    perform_separation(path)

for entry in os.scandir(directory):
    if entry.is_file() and regWav.match(entry.path):
        perform_text_recognition(entry.path)

# ШУМОПОДАВЛЕНИЕ

noisePaths = []

for entry in os.scandir(directory):
    if entry.is_file() and regNoise.match(entry.path):
        perform_noise_reduction(entry.path)

for entry in os.scandir(directory):
    if entry.is_file() and regNoiseRed.match(entry.path):
        perform_diarization(entry.path, os.getenv(token_env_key))
        perform_separation(entry.path)

for entry in os.scandir(directory):
    if entry.is_file() and regNoiseRedSep.match(entry.path):
        perform_text_recognition(entry.path)

