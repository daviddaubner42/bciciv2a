import mne
import os
import pickle
from pathlib import Path
from mne.preprocessing import EOGRegression
from mne import pick_types

def load_raw(raw_path, save=True):
    raw = mne.io.read_raw_gdf(raw_path, preload=True)
    raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
    if save:
        os.makedirs("raw", exist_ok=True)
        subject_id = raw.info['subject_info']['his_id']
        raw.save(Path("raw", f"{subject_id}_raw.fif"), overwrite=True)

def regress_eog(raw, weights=None, save=True):
    ref_raw = mne.set_eeg_reference(raw, 'average', projection=False)[0]
    filt_raw = ref_raw.filter(l_freq=1, h_freq=None)

    if weights == None:
        weights = EOGRegression(picks="eeg", picks_artifact="eog").fit(raw)

        subject_id = raw.info['subject_info']['his_id']

        if save:
            with open(Path("eog_regression", f"{subject_id}_regr_weights.pkl"), "wb") as f:
                pickle.dump(weights, f)

    regr_raw = weights.apply(raw, copy=True)

    if save:
        regr_raw.save(Path("preproc", f"{subject_id}_regr_raw.fif"), overwrite=True)

    return regr_raw, weights

def epoch(raw, tmin=0, tmax=4):
    # Bandpass filter tu mu/beta band
    # (Most discriminatory signal in motor imagery is there)
    filt_raw = raw.filter(l_freq=8, h_freq=30)

    events, event_id = mne.events_from_annotations(filt_raw)
    picks = pick_types(filt_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(filt_raw, events, event_id=event_id, picks=picks, tmin=tmin, tmax=tmax, baseline=None, preload=True, event_repeated='drop')
    return epochs

def preprocess(raw_path, weights=None, save=True):
    raw = load_raw(raw_path, save)
    regr_raw, weights = regress_eog(raw, weights, save)
    epochs = epoch(regr_raw, tmin=0, tmax=4)
    return epochs