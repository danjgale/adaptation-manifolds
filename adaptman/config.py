"""Configure file paths and parameters for project


"""
import os
pjoin = os.path.join

class Config(object):

    # project directories
    config_path = os.path.dirname(os.path.abspath(__file__))
    resources = pjoin(config_path, '../resources')
    fmriprep_dir = '/Raid6/raw/VMR-Learning-Complete/derivatives/2020/fmriprep'

    # parcellation
    atlas = pjoin(resources, 'atlases',
                  'Schaefer2018_1000Parcels_7Networks_order.dlabel.nii')
    adjacency = pjoin(resources, 'atlases', 
                      'Schaefer2018_1000Parcels_7Networks_adjacency.tsv')
    
    # data directories
    data_dir = pjoin(config_path, '../data')
    dataset = 'schaefer1000-7networks-final'
    dataset_dir = pjoin(data_dir, dataset)
    tseries = pjoin(dataset_dir, 'timeseries')
    connect = pjoin(dataset_dir, 'connectivity')

    # paths expected to change
    gradients = pjoin(dataset_dir, 'pca-gradients-centered-ses-01')

    k = 3
    results = pjoin(config_path, f'../results/k{k}')
    figures = pjoin(config_path, f'../figures/fig-components-pngs-k{k}')

    os.makedirs(results, exist_ok=True)
    os.makedirs(figures, exist_ok=True)
