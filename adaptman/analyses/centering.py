"""Distance analyis showing power of centering procedure"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.utils.distance import distance
from umap import UMAP

from adaptman.config import Config
from adaptman.utils import get_files
from adaptman.analyses.plotting import set_plotting


set_plotting()


def load_matrices(fnames):

    data, metadata = [], []
    for i in fnames:
        basename = os.path.basename(i) 
        metadata.append({
            'sub': basename[:6], 
            'epoch': basename.split('.')[0].split('_')[-1]
        })
        data.append(pd.read_table(i, index_col=0).values)
    return data, pd.DataFrame(metadata)


def pairwise_distances(matrices):

    n_matrices = len(matrices)
    indices = np.arange(len(matrices))

    dmat = np.zeros((n_matrices, n_matrices))
    pairs = itertools.combinations(indices, 2)
    for p in pairs:
        i, j = p
        val = distance(matrices[i], matrices[j])
        dmat[i, j] = val
        dmat[j, i] = val

    return dmat


def embed_data(fnames):
    data, metadata = load_matrices(fnames)
    dmat = pairwise_distances(data)

    reducer = UMAP(metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(dmat)
    
    return pd.concat([metadata, pd.DataFrame(embedding)], axis=1)


def embedding_plot(data, out_dir):

    for d, name in zip(data, ['uncentered', 'centered']):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.5, 2.5), 
                                 sharey=True, sharex=True)
        kwargs = dict(legend=False, s=10, clip_on=False)
        sns.scatterplot(x=0, y=1, hue='sub', data=d, ax=axes[0], **kwargs)
        sns.scatterplot(x=0, y=1, hue='epoch', data=d, ax=axes[1], 
                        palette='Dark2', **kwargs)
        for ax in axes:
            ax.set(xlabel='Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[1].set_ylabel('')
        # axes[1].set_yticklabels([])

        sns.despine()
        fig.suptitle(name.title(), size=10)
        fig.tight_layout()
        
        fig.savefig(os.path.join(out_dir, f'{name}_umap'), dpi=300)


def main():
    config = Config()
    original = get_files([config.connect, 'sub*/*ses-01*rotation*.tsv'])
    centered = get_files([config.dataset_dir, 
                          'connectivity-centered-ses-01/sub*/*rotation*.tsv'])

    datasets = []
    for i in [original, centered]:
        datasets.append(embed_data(i))

    fig = embedding_plot(datasets, config.figures)
    fig.savefig(os.path.join(config.figures, 'centering_umap'), dpi=300)


if __name__ == '__main__':
    main()
