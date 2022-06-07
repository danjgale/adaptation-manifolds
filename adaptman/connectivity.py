"""Compute connectivity matrices for each timeseries"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import sqrtm, logm, expm, invsqrtm

from adaptman.config import Config
from adaptman.utils import get_files, display

pjoin = os.path.join


def _baseline_window_length():
    """Number of baseline scans with discarded scans + non-task scans 
    accounted for
    """
    return 240


def _drop_nontask_samps(x):
    """Remove samples before and after the task"""
    return x[4:-8, :]


def _split_by_learning(x):
    """Divide rotation scan into baseline, early learning, and late learning"""
    window_length = _baseline_window_length()

    baseline = x[:window_length]
    early = x[window_length:window_length * 2]
    late = x[-window_length:]
    return [baseline, early, late]


def _split_by_rotation(x):
    """Divide rotation scan into baseline (rotation-off) and learning 
    (rotation-on)
    """
    window_length = _baseline_window_length()

    baseline = x[:window_length]
    learning = x[window_length:]
    return [baseline, learning]


def compute_connectivity(timeseries, output_dir, float_fmt='%1.8f', 
                         split='learning'):
    """Run a connectivity analysis within a single timeseries or between two
    timeseries. 

    Computes the pairwise correlations between each column in the timeseries 
    file(s). Saves the resulting correlation matrix to the output directory;
    the save filename is the same as the input filename (therefore, do NOT save
    to the same directory as the input data) in a subject-specific folder. If 
    two timeseries are provided, only the connectivity _between_ the timeseries 
    are provided (i.e. cross connectivity).  

    If the data belongs to rotation scans, they will be split into baseline, 
    early, and late epochs and saved as such.

    Parameters
    ----------
    timeseries : str
        Timeseries file name
    output_dir : str
        Top-level output directory
    float_fmt : str, optional
        Float format for output. Can save space if connectivity matrices are 
        very large. Note that correctly rounds the data rather than plainly 
        truncating floating points. By default '%1.8f'
    """
    fname = os.path.split(timeseries)[1]
    display(fname)
    data = pd.read_table(timeseries)
    tseries = data.values
    regions = data.columns

    # handle different scan types
    if 'rotation' in fname:
        tseries = _drop_nontask_samps(tseries)
        
        if split == 'learning':
            dataset = _split_by_learning(tseries)
        elif split == 'rotation':
            dataset = _split_by_rotation(tseries)
        else:
            dataset = [tseries]

    elif 'washout' in fname:
        dataset = [_drop_nontask_samps(tseries)]
    else:
        dataset = [tseries]

    conn = ConnectivityMeasure(kind='covariance')
    connectivities = conn.fit_transform(dataset)

    # set up subject-specific outputs
    output_dir = pjoin(output_dir, fname[:6])
    os.makedirs(output_dir, exist_ok=True)
    
    # save
    cmat_name = fname.split('_space')[0] + '_cmat'
    n_matrices = len(connectivities)
    if n_matrices == 2:
        suffix = ['_base', '_learn']
    elif n_matrices == 3:
        suffix = ['_base', '_early', '_late']
    else:
        suffix = ['']
        
    for s, cmat in zip(suffix, connectivities):
        out = pjoin(output_dir, cmat_name + s + '.tsv')
        out_cmat = pd.DataFrame(cmat, index=regions, columns=regions)
        out_cmat.to_csv(out, float_format=float_fmt, sep='\t')


def connectivity_analysis(input_data, out_dir, ses=None, njobs=45):
    """Full connectivity analysis on provided data

    Parameters
    ----------
    input_data : str
        Path input data folder. 
    out_dir : str
        Path to output directory.
    njobs : int
        Number of processes to run in parallel. Default is 32.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if ses is not None:
        timeseries = get_files(input_data + f'/*{ses}*.tsv')
    else:
        timeseries = get_files(input_data + '/*.tsv')
    # get only task data
    timeseries = [i for i in timeseries if 'rest' not in i]
    Parallel(njobs)(delayed(compute_connectivity)(ts, out_dir) 
        for ts in timeseries
    )

# Covariance centering
def _to_tangent(s, mean):
    p = sqrtm(mean)
    p_inv = invsqrtm(mean)
    return p @ logm(p_inv @ s @ p_inv) @ p 
    

def _gl_transport(t, sub_mean, grand_mean):
    g = sqrtm(grand_mean) @ invsqrtm(sub_mean)
    return g @ t @ g.T


def _from_tangent(t, grand_mean):
    p = sqrtm(grand_mean)
    p_inv = invsqrtm(grand_mean)
    return p @ expm(p_inv @ t @ p_inv) @ p 


def center_cmat(c, sub_mean, grand_mean):
    """Center covariance matrix using tangent transporting procedure

    Parameters
    ----------
    c : numpy.ndarray
        Single MxM covariance matrix of a single subject
    sub_mean : numpy.ndarray
        Geometric mean covariance matrix of the subject   
    grand_mean : numpy.ndarray
        Geometric mean across all subjects and matrices

    Returns
    -------
    numpy.ndarray
        Centered covariance matrix
    """
    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)


def center_subject(sub_cmats, grand_mean):
    """Center all of a subject's covariance matrices with respect to the 
    grand geometric mean

    Parameters
    ----------
    sub_cmats : list of numpy.ndarray
        List of subject's NxN covariance matrices
    grand_mean : numpy.ndarray
        Geometric mean across all subjects and matrices

    Returns
    -------
    numpy.ndarray
        Covariance matrices in shape K x M x M, where K is the number/length of 
        sub_cmats and M is the dimensions of the covariance matrix. Matrices
        are in the same order as sub_cmats
    """
    sub_mean = mean_riemann(sub_cmats)
    return np.array([center_cmat(c, sub_mean, grand_mean) for c in sub_cmats])


def _read_and_stack_cmats(x):
    """Load connectivity matrix and create a single array with dims 
    N cmats x M rois x M rois
    """
    arr = np.array([pd.read_table(i, index_col=0).values for i in x])
    labels = pd.read_table(x[0], index_col=0).columns
    return arr, labels


def center_matrices(dataset_dir, ses=None, float_fmt='%1.8f'):
    """Center all covariance matrices according to tangent transport centering
    procedure

    Parameters
    ----------
    dataset_dir : str
        Top-level directory containing all subjects' connectivity matrices
    ses : str, optional
        Specify a specific session, either 'ses-01' or 'ses-2'; by default None
    float_fmt : str, optional
        Output float format. Note that correctly rounds the data rather than
        plainly truncating floating points. By default '%1.8f'
    """
    if ses is None:
        cmats = get_files([dataset_dir, '*/*.tsv'])
        out_dir = dataset_dir + '-centered'
    else:
        cmats = get_files([dataset_dir, f'*/*{ses}*.tsv'])
        out_dir = dataset_dir + f'-centered-{ses}'

    os.makedirs(out_dir, exist_ok=True)
    
    # compute geometric mean covariance matrix
    all_cmats, roi_labels = _read_and_stack_cmats(cmats)
    display('Computing grand mean')
    grand_mean = mean_riemann(all_cmats)

    # save grand mean
    df = pd.DataFrame(grand_mean, index=roi_labels, columns=roi_labels) 
    df.to_csv(pjoin(out_dir, 'grand_mean.tsv'), sep='\t', 
              float_format=float_fmt)

    subs = np.unique([os.path.basename(x)[:6] for x in cmats])
    for s in subs:
        display(s)
        sub_files = [i for i in cmats if s in i]
        sub_cmats, _ = _read_and_stack_cmats(sub_files)
        centered_cmats = center_subject(sub_cmats, grand_mean)

        for i, fname in enumerate(sub_files):
            df = pd.DataFrame(centered_cmats[i], index=roi_labels, 
                              columns=roi_labels) 

            out_name = fname.replace(dataset_dir, out_dir)
            os.makedirs(os.path.split(out_name)[0], exist_ok=True)
            df.to_csv(out_name, sep='\t', float_format=float_fmt)


def main():

    config = Config()
    connectivity_analysis(config.tseries, config.connect)
    center_matrices(config.connect, ses='ses-01')


if __name__ == "__main__":
    main()

