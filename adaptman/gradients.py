"""Compute connectivity gradients

Three different types of gradients are computed: 
1 - bilateral within-PCG gradients
2 - bilateral PCG-cortex gradients using Schaefer 400 (SomMot excluded)
3 - unilateral PCG-cortex gradients using Schaefer 400 (SomMot excluded)
"""
import os
import re
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from pyriemann.utils.mean import mean_riemann

from adaptman.config import Config
from adaptman.utils import get_files, display


def _pca_gradients(x, ref=None):
    """PCA model"""
    alignment = None if ref is None else 'procrustes'
    gm = GradientMaps(n_components=None, approach='pca', kernel='cosine', 
                      alignment=alignment)
    gm.fit(x, reference=ref)
    return gm


def _dm_gradients(x, ref=None):
    """Diffusion embedding model"""
    alignment = None if ref is None else 'procrustes'
    gm = GradientMaps(n_components=20, approach='dm', kernel='cosine',
                      alignment=alignment, random_state=42)
    gm.fit(x, diffusion_time=0, reference=ref)
    return gm


def save_gradients(gradients, lambdas, regions, out_prefix, float_fmt=None, 
                   save_threshold=.01):
    """Save gradients and eigenvalues

    Parameters
    ----------
    gradients : numpy.ndarray
        Gradient array
    lambdas : np.ndarray
        Eigenvalues associated with gradients
    regions : list
        Region labels
    out_prefix : str
        Prefix for output files. Should include output directory
    float_fmt : str, optional
        Format to reduce the number of decimal points when saving gradients. 
        Use to reduce file sizes of output. By default None
    save_threshold : float, optional
        Only save gradients/components above a proportion explained threshold.
        Note that all gradient eigenvalues will be saved to the eigenvalue file
        regardless. This will drastically reduce the file size of the gradient 
        file. By default .01
    """
    n_gradients = gradients.shape[1]
    labels = [f'g{x}' for x in np.arange(n_gradients) + 1]
    
    gradients = pd.DataFrame(gradients, index=regions, columns=labels)
    eigens = pd.DataFrame({'eigenvalues': lambdas, 
                           'proportion': lambdas / lambdas.sum()}, 
                           index=labels)

    if save_threshold is not None:
        retain = eigens[eigens['proportion'] > save_threshold].index.values
        gradients = gradients[retain]

    gradients.to_csv(out_prefix + '_gradient.tsv', sep='\t', 
                     float_format=float_fmt)
    eigens.to_csv(out_prefix + '_eigenvalues.tsv', sep='\t', 
                   float_format=float_fmt)


def epoch_gradients(cmat_file, reference, out_dir, approach='pca'):
    """Compute gradients for a single experiment epoch, aligned to a given 
    reference gradient

    Parameters
    ----------
    cmat_file : str
        Path to a connectivity matrix file produced in connectivity.py
    reference : numpy.ndarray
        Pre-computed reference gradient
    out_dir : str
        Output directory. Subject ID from `cmat_file` will automatically be
        appended
    approach : str
        Dimensionality reduction approach, either 'dm' or 'pca' (default)
    """
    fname = os.path.split(cmat_file)[1]
    sub_id = fname[:6]

    display(f'{fname}')
    cmat = pd.read_table(cmat_file, index_col=0)
    regions = cmat.columns

    if approach == 'dm':
        gm = _dm_gradients(cmat.values, reference)
    elif approach == 'pca':
        gm = _pca_gradients(cmat.values, reference)

    # save gradients
    save_dir = os.path.join(out_dir, sub_id)
    os.makedirs(save_dir, exist_ok=True)
    out_prefix = os.path.join(save_dir, fname.replace('_cmat', '')[:-4])
    save_gradients(gm.aligned_, gm.lambdas_, regions, out_prefix)
    

def dataset_gradient(cmats, out_dir, reference, approach='pca', n_jobs=32):
    """Run gradient calculations on a provided connectivity dataset.

    The resulting gradients are saved in subject-specific directories. In 
    addition, the reference gradients are also saved.

    Parameters
    ----------
    cmats : list
        List of connectivity matrix files
    out_dir : str
        Path to top-level output directory. Subject-specific folders will be 
        automatically generated. 
    reference : str, optional
        Path to connectivity matrix used as reference
    approach : str
        Dimensionality reduction approach, either 'dm' or 'pca' (default).  
    """
    os.makedirs(out_dir, exist_ok=True)

    # generate reference gradient
    ref_cmat = pd.read_table(reference, index_col=0)
    if approach == 'dm':
        gm = _dm_gradients(ref_cmat.values)
    elif approach == 'pca':
        gm = _pca_gradients(ref_cmat.values)
    out_prefix = os.path.join(out_dir, 'reference')
    save_gradients(gm.gradients_, gm.lambdas_, ref_cmat.columns, out_prefix)

    # epoch gradients
    Parallel(n_jobs)(delayed(epoch_gradients)(x, gm.gradients_, out_dir, 
                                              approach) 
        for x in cmats
    )

def shuffle_labels(files, out_dir, float_fmt='%1.8f', verbose=False):

    sub_ids = np.unique([re.findall(r'sub-\d+', f)[0] for f in files])
    mappings = []
    for i in sub_ids:

        sub_dir = os.path.join(out_dir, i)
        os.makedirs(sub_dir, exist_ok=True)

        fnames = [f for f in files if i in f]

        indices = list(range(len(fnames)))
        np.random.shuffle(indices)

        for j, k in zip(indices, fnames):
            cmat = pd.read_table(k, index_col=0)
            if verbose:
                print(k, '->', fnames[j])

            out_basename = os.path.basename(fnames[j])
            cmat.to_csv(os.path.join(sub_dir, out_basename), sep='\t', 
                        float_format=float_fmt)
            mappings.append(
                {'original': os.path.basename(k), 'new': out_basename}
            )

    df = pd.DataFrame(mappings)
    df.to_csv(os.path.join(out_dir, 'mappings.csv'), index=False)


def create_reference(dataset_dir, ses='ses-01', specifier='base', 
                     mean='arithmetic'):

    if ses == 'both':
        pattern = f'*/*{specifier}*.tsv'
    else:
        pattern = f'*/*{ses}*{specifier}*.tsv'
    cmat_files = get_files([dataset_dir, pattern])
    
    cmats = np.array([pd.read_table(i, index_col=0).values for i in cmat_files])
    labels = pd.read_table(cmat_files[0], index_col=0).columns

    if mean == 'arithmetic':
        mean_cmat = np.mean(cmats, axis=0)
    elif mean == 'geometric':
        mean_cmat = mean_riemann(cmats)
    else:
        raise ValueError("mean must be 'arithmetic' or 'geometric'")

    df = pd.DataFrame(mean_cmat, index=labels, columns=labels)
    out = os.path.join(dataset_dir, 'reference_cmat.tsv')
    df.to_csv(out, sep='\t')
    return out


def main():
    config = Config()

    suffix = '-centered-ses-01'
    cmat_files = get_files([config.connect + suffix, '*/*ses-01*.tsv'])
    
    ref_cmat = create_reference(config.connect + suffix, 
                                specifier='base', mean='geometric')
    
    out_dir = os.path.join(config.dataset_dir, 
                        f'pca-gradients{suffix}')
    dataset_gradient(cmat_files, out_dir, ref_cmat)


if __name__ == "__main__":
    main()
    
