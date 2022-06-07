"""Supplementary washout analysis"""
import os
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from surfplot import Plot

from adaptman.config import Config
from adaptman.utils import get_surfaces, fdr_correct, test_regions
from adaptman.analyses import plotting

FIG_DIR = os.path.join(Config().figures, 'washout')
os.makedirs(FIG_DIR, exist_ok=True)

plotting.set_plotting()


def mean_stat_map(data, out_dir, mean_center=False):
    """Plot mean eccentricity of each region

    Parameters
    ----------
    data : pd.DataFrame
        Subject gradients data
    out_dir : str
        Output file path
    mean_center : bool, optional
        Mean center each region, by default False

    Returns
    -------
    matplotlib.pyplot.figure
        Brain surface figure of mean eccentricity
    """
    # mean across all epochs
    mean_ecc = data.groupby(['epoch', 'roi', 'roi_ix'])['distance'] \
                   .mean() \
                   .reset_index() \
                   .sort_values(['epoch', 'roi_ix'])
    
    # roi by epochs
    epoch_ecc = pd.DataFrame({name: g['distance'].values 
                              for name, g in mean_ecc.groupby('epoch')})
    adapt_ecc = epoch_ecc[['base', 'early', 'late']]
    washout_ecc = epoch_ecc['washout']

    if mean_center:

        # subtract mean of adaptation epochs; i.e. washout is relative to 
        # adaptation
        roi_means = epoch_ecc.mean(axis=1)
        washout_ecc -= roi_means

        cmap = sns.diverging_palette(210, 5, s=100, l=30, sep=10, 
                                     as_cmap=True)

        # match colorscale to adaptation color scale, which requires centering 
        # the adaptation epochs
        adapt_ecc = adapt_ecc.apply(lambda x: x - x.mean(), axis=1)
        vmax = np.around(np.nanmax(adapt_ecc), 2)
        # vmax = .5
        vmin = -vmax
        n_ticks = 3
        prefix = 'centered_mean_ecc_'
    else:
        cmap = 'viridis'
        vmax = np.nanmax(adapt_ecc)
        vmin = np.nanmin(adapt_ecc)
        n_ticks = 2
        prefix = 'mean_ecc_'

    prefix = os.path.join(out_dir, prefix)
    plotting.plot_cbar(cmap, vmin, vmax, 'horizontal', size=(.8, .25), 
                         n_ticks=2)
    plt.savefig(prefix + 'cbar') 

    config = Config()
    surfaces = get_surfaces()

    x = plotting.weights_to_vertices(washout_ecc, config.atlas)
    p = Plot(surfaces['lh'], surfaces['rh'], layout='row', 
                mirror_views=True, size=(800, 200), zoom=1.2)
    p.add_layer(x, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    fig = p.build()
    fig.savefig(prefix + 'washout_brain')

    return fig


def compare_washout(data):
    """Perform paired t-tests between washout and adaptation epochs

    Parameters
    ----------
    data : pd.DataFrame
        Subject gradients data

    Returns
    -------
    pd.DataFrame
        Pairwise t-test results
    """
    washout = data.query("epoch == 'washout'")
    
    list_ = []
    adapt_epochs = ['base', 'early', 'late']
    for epoch in adapt_epochs:
        df = pd.concat([washout, data.query("epoch == @epoch")])
        df = df[['sub', 'epoch', 'roi', 'roi_ix', 'distance']]

        res = test_regions(df, 'ttest')
        list_.append(res)
    comparisons = pd.concat(list_)
    comparisons = fdr_correct(comparisons)

    return comparisons


def embedding_similarity(data, anova, out_dir, draw_errbar=True, 
                         draw_line=True):
    """Embed adaptation and washout epochs using regions that show significant
    changes during adaptation

    This analysis situates washout with respect to adapation, using 
    learning-relevant. UMAP is used to perform low-dimensional embedding. 

    Parameters
    ----------
    data : pd.DataFrame
        Subject gradients data
    anova : pd.DataFrame
        Adaptation ANOVA stats table 
    out_dir : str
        File path for output figure
    draw_errbar : bool, optional
        Draw 2D error bars around mean, by default True
    draw_line : bool, optional
        Draw temporal trajectory that connects each epoch in sequential order, 
        by default True
    """
    df = data.loc[:, ['roi', 'sub', 'epoch', 'distance']] \
             .pivot(['sub', 'epoch'], 'roi', 'distance')
    
    if isinstance(anova, str):
        anova = pd.read_table(anova)
    
    sig_regions = anova.query("sig_corrected == 1")['roi'].tolist()

    df = df[sig_regions]

    mds = UMAP(random_state=42)
    x = StandardScaler().fit_transform(df.values)
    loadings = pd.DataFrame(mds.fit_transform(x))
    loadings = loadings.rename(columns={0: 'dim1', 1: 'dim2'})
    loadings['sub'] = df.index.get_level_values(0).values
    loadings['epoch'] = df.index.get_level_values(1).values

    means = loadings.groupby('epoch').mean()
    stderr = loadings.groupby('epoch').std() / np.sqrt(32)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    cmap = sns.color_palette("Set2", 4)
    ax.set_prop_cycle(color=cmap)
    for epoch in means.index.values:
        x = means.loc[epoch, 'dim1']
        y = means.loc[epoch, 'dim2']
        if draw_errbar:
            ax.errorbar(x, y, stderr.loc[epoch, 'dim1'], 
                        stderr.loc[epoch, 'dim2'], lw=2)
        ax.scatter(x, y, edgecolor='k', marker='s', zorder=2, s=40, 
                   linewidths=1)

    if draw_line:
        ax.plot(means.values[:, 0], means.values[:, 1], c='k', 
                zorder=-1, lw=1)

    if not draw_errbar:
        ax = sns.scatterplot(x='dim1', y='dim2', hue='epoch', 
                             data=loadings, s=20, palette=cmap)
    sns.despine()
    # ax.set(xlabel='Dimension 1', ylabel='Dimension 2')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set(xticks=np.arange(0, 5, 2))
    fig.tight_layout()

    out_file = os.path.join(out_dir, 'washout_embedding_similarity.png')
    fig.savefig(out_file)


def main():
    config = Config()
    data = pd.read_table(os.path.join(config.results, 'subject_gradients.tsv'))
    mean_stat_map(data, FIG_DIR)
    mean_stat_map(data, FIG_DIR, mean_center=True)

    df = compare_washout(data)
    plotting.pairwise_stat_maps(df, os.path.join(FIG_DIR, 'ecc_ttests_'), 
                                vmin=2.29, vmax=5.72, 
                                cbar_orientation='horizontal')

    anova = os.path.join(config.results, 'ecc_anova_stats.tsv')
    res = embedding_similarity(data, anova, FIG_DIR)


if __name__ == '__main__':
    main()
