import librosa
import pyAudioAnalysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as af

import glob, os
import numpy as np
import pandas as pd
import parselmouth 
import statistics


from parselmouth.praat import call, run_file
from scipy.stats.mstats import zscore
import IPython 

import swifter
import argparse
import pandas as pd
from audio_features import*

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", "-pr", help="Mention prompt.")
parser.add_argument("--start_index", "-si",type = int, help="Start index of dataframe.", default = 0)
parser.add_argument("--end_index", "-ei", type =int, help="End index of dataframe.", default = 100000)

args = parser.parse_args()
PROMPT_TYPE = args.prompt
START = args.start_index
END = args.end_index
DATA_DIR = f"/media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/{PROMPT_TYPE}/"
SAV_DIR = f"/media/nas_mount/Sarthak/Pakhi/data/{PROMPT_TYPE}/" 

ALL_AUDIOS = glob.glob(f'{DATA_DIR}/*.wav')

feature_columns = ["duration","stdev_energy", "mean_pitch", "stdev_pitch", "range_pitch",
                  "voiced_to_total_ratio","voiced_to_unvoiced_ratio","zrc","energy","energy_entropy",
                   "spectral_centroid","spectral_entropy","localJitter","localabsoluteJitter","rapJitter",
                   "ppq5Jitter","ddpJitter","localShimmer","apq3Shimmer","aqpq5Shimmer","apq11Shimmer",
                   "ddaShimmer","f1_mean","f2_mean","f3_mean","f4_mean","f1_median","f2_median","f3_median","f4_median"]

df = pd.DataFrame([], columns = feature_columns)
df['name'] = ALL_AUDIOS
df.to_csv("f'{SAV_DIR}/{PROMPT_TYPE}_all_audio_features{START}_{END}.csv'

df[feature_columns] = df['name'].swifter.apply(lambda x: main(x))

df.to_csv(f'{SAV_DIR}/{PROMPT_TYPE}_all_audio_features{START}_{END}.csv')