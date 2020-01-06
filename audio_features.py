import librosa
import pyAudioAnalysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as af
import time

import glob, os
import numpy as np
import pandas as pd
import parselmouth 
import statistics


from parselmouth.praat import call, run_file
from scipy.stats.mstats import zscore
import IPython 

import swifter
import textgrids
import argparse
from fluency import*
import pandas as pd

def get_duration(y,sr):
    """
    Gets duration of audio via librosa.

    Parameters
    ----------
    
    y:np.ndarray[shape = (n,) or (2,n)] 
        audio time series
    sr : number>0 [scalar]
        sampling rate of y

    Returns
    -------
    float
        Duration of audio sample in seconds.

    Examples
    --------
    >>> get_duration(y, sr)
    59.78
    """
    return librosa.get_duration(y=y, sr=sr)

def get_stdev_energy(y,sr):
    """
    Gets rms energy for each audio frame of audio via librosa.
    Takes standard deviation of the obtained ndarray.

    Parameters
    ----------
    y:np.ndarray[shape = (n,) or (2,n)] 
        audio time series
        
    Returns
    -------
    float
        Standard deviation of rms energy for each frame of the audio sample.

    Examples
    --------
    >>> get_stdev_energy(y)
    59.78
    """
    energy = librosa.feature.rms(y=y)
    
    return np.std(energy)

def get_pitch(sound, _mean = True, _stdev= False, _range = False):
    """
    Gets pitch for each audio frame of audio via parselmouth praat.
    Takes mean or range of the obtained ndarray.

    Parameters
    ----------
    sound:parselmouth object
        audio object
    _stdev:boolean
        True, if want to get standard deviation of pitch
    _range:boolean
        True, if want to get range of deviation of pitch        
        
    Returns
    -------
    float
        Mean pitch of the audio sample.

    Examples
    --------
    >>> get_stdev_energy(y)
    59.78
    """
    pitch = call(sound, "To Pitch", 0.0, 75, 300)
    if _mean:
        mean_pitch = call(pitch, "Get mean", 0, 0,'Hertz')
        return mean_pitch
    if _stdev:
        stdev_pitch = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        return stdev_pitch
    if _range:
        stdevPitch = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        range_pitch = 4* stdevPitch
        return range_pitch
    
def get_voiced_frames(sound, vtt = True, vtu = False):
    """
    Gets ratio of voiced to total audio frames or
    ratio of voiced to unvoiced frames.
    
    Parameters
    ----------
    sound:parselmouth object
        audio object
    vtt:boolean
        if True returns ratio of voiced frames to total frames.
    vtu:boolean
        if True returns ratio of voiced frames to unvoiced frames.      
        
    Returns
    -------
    float
        Ratio.

    Examples
    --------
    >>> get_stdev_energy(y)
    59.78
    """
    pitch = call(sound, "To Pitch", 0.0, 75, 300)
    voiced_frames = pitch.count_voiced_frames()
    total_frames = pitch.get_number_of_frames()
    
    if vtt:
        voiced_to_total_ratio = voiced_frames/total_frames
        return voiced_to_total_ratio
    if vtu:
        voiced_to_unvoiced_ratio =  voiced_frames / (total_frames - voiced_frames)
        return voiced_to_unvoiced_ratio
    
    
def get_Zero_Crossing_Rate(y):
    '''
    Zero Crossing Rate - The rate of sign-changes of the signal during the duration of a particular frame.
    '''   
    return af.stZCR(y)

def get_Energy(y):
    '''
    Energy- The sum of squares of the signal values, normalized by the respective frame length.
    '''
    return af.stEnergy(y)

def get_Energy_Entropy(y):
    '''
    Entropy of Energy - The entropy of sub-frames normalized energies. It can be interpreted as a measure of abrupt changes.
    '''
    return af.stEnergyEntropy(y)

def get_Spectral_Centroid_And_Spread(y,sr):
    '''
    Spectral Centroid - The center of gravity of the spectrum.
    Spectral Spread - The second central moment of the spectrum.
    '''
    return af.stSpectralCentroidAndSpread(y, sr)

def get_Spectral_Entropy(y):
    '''
    Spectral Entropy - Entropy of the normalized spectral energies for a set of sub-frames.
    '''
    return af.stSpectralEntropy(y)

# def get_zrc(librosa_val):
#     y, sr = librosa_val
#     zrc = get_Zero_Crossing_Rate(y)
#     return zrc

# def get_Energy(librosa_val):
#     y, sr = librosa_val
#     energy = get_Energy(y)
#     return energy

def get_spectral_centroid(y,sr):
    spectral_centroid, _= get_Spectral_Centroid_And_Spread(y,sr)
    return spectral_centroid


def get_hnr_shimmer_jitter(sound,f0min = 75, f0max = 300, unit ="Hertz"):   
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter= call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer= call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer= call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer=  call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer= call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer=call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer=call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer

# This function measures formants using Formant Position formula
def measureFormants(sound, f0min=75,f0max=300):
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    try:
        f1_mean=  statistics.mean(f1_list)
        f2_mean= statistics.mean(f2_list)
        f3_mean=statistics.mean(f3_list)
        f4_mean= statistics.mean(f4_list)
        f1_median=statistics.median(f1_list)
        f2_median= statistics.median(f2_list)
        f3_median= statistics.median(f3_list)
        f4_median= statistics.median(f4_list)
        return f1_mean,f2_mean,f3_mean,f4_mean,f1_median,f2_median,f3_median,f4_median
    except:
        return None,None,None,None,None,None,None,None,
    
def main(audpath):
    sound = parselmouth.Sound(audpath) 
    y, sr = librosa.core.load(audpath)
#     start_time = time.time()
    duration = get_duration(y,sr)
#     print("Duration--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    stdev_energy = get_stdev_energy(y,sr)
#     print("stdev_energy--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    mean_pitch = get_pitch(sound, _mean = True, _stdev= False, _range = False)
#     print("mean_pitch--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    stdev_pitch = get_pitch(sound, _mean = False, _stdev= True, _range = False)
#     print("stdev_pitch--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    range_pitch = get_pitch(sound, _mean = False, _stdev= False, _range = True)
#     print("range_pitch--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    voiced_to_total_ratio = get_voiced_frames(sound, vtt = True, vtu = False)
#     print("voiced_to_total_ratio--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    voiced_to_unvoiced_ratio = get_voiced_frames(sound, vtt = False, vtu = True)
#     print("voiced_to_unvoiced_ratio--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    zrc = get_Zero_Crossing_Rate(y)
#     print("get_Zero_Crossing_Rate--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    energy = get_Energy(y)
#     print("get_Energy--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    energy_entropy = get_Energy_Entropy(y)
#     print("energy_entropy--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    spectral_centroid = get_spectral_centroid(y,sr)
#     print("spectral_centroid--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    spectral_entropy = get_Spectral_Entropy(y)
#     print("spectral_entropy--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer= get_hnr_shimmer_jitter(sound,f0min = 75, f0max = 300, unit ="Hertz")
#     print("hnr_jitter_shimmer--- %s seconds ---" % (time.time() - start_time))
#     start_time = time.time()
    f1_mean,f2_mean,f3_mean,f4_mean,f1_median,f2_median,f3_median,f4_median = measureFormants(sound, f0min=75,f0max=300)
#     print("formants--- %s seconds ---" % (time.time() - start_time))
    return pd.Series([duration, stdev_energy, mean_pitch, stdev_pitch, range_pitch,voiced_to_total_ratio,voiced_to_unvoiced_ratio,zrc,energy,energy_entropy,spectral_centroid,spectral_entropy,localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer,f1_mean,f2_mean,f3_mean,f4_mean,f1_median,f2_median,f3_median,f4_median])

    
    
# import pandas as pd
# import swifter

# df_t = pd.read_csv(f'{DATA_FILE}/{QUES_TYPE}/{FILE_TYPE}.csv')
# df_t['spectral_entropy'] = df_t['text'].swifter.apply(lambda x: get_syllable_count(x))
    