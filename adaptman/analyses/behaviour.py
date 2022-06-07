"""Behaviour analyses"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from surfplot import Plot
import cmasher as cmr

from adaptman.config import Config
from adaptman.utils import get_surfaces, parse_roi_names, fdr_correct
from adaptman.analyses import plotting

plotting.set_plotting()

FIG_DIR = os.path.join(Config().figures, 'behaviour')
os.makedirs(FIG_DIR, exist_ok=True)


def _label_trial_blocks(df, trial_block_length=8):
    """Enumerate trial blocks"""
    n_trials = df['Trial'].max()
    n_blocks = n_trials / trial_block_length
    df['trial_block'] = np.repeat(np.arange(n_blocks) + 1, trial_block_length)
    return df


def bin_data(data):
    """Label and bin data according to trial blocks

    Parameters
    ----------
    data : pd.DataFrame
        Original trial-by-trial data

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Labeled data and binned measures
    """
    group_vars = ['Day', 'Number', 'Block']
    df = data.copy().groupby(group_vars).apply(_label_trial_blocks)

    bin_measures = df.groupby(group_vars + ['trial_block'])['Error_deg'] \
                     .describe() \
                     .reset_index()
    return df, bin_measures


def task_behaviour_plot(data):
    """Plot group-average error throughout the entirety of the task

    Parameters
    ----------
    data : pd.DataFrame
        Subject-level, trial-wise data
    """
    df = data.query("Day == 'Day 1' and Block != 'Washout'")
    blocks_per_epoch = 15
    trials_per_block = 8
    trials_per_epoch = trials_per_block * blocks_per_epoch

    # add baseline trial count to rotation
    df.loc[df['Block'] == 'Rotation', 'Trial'] += trials_per_epoch
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax = sns.lineplot(x='Trial', y='Error_deg', data=df, ci=68, 
                      color='slategray', ax=ax, zorder=3)
    plt.setp(ax.collections[0], alpha=0.2)

    max_trial = df['Trial'].max()
    xticks = np.arange(0, max_trial + 20, 20)
    ax.set(xticks=xticks, yticks=np.arange(-30, 75, 15),
           xlim=(1, max_trial), ylabel='Angular error (°)', 
           xlabel='Trial')

    ax.set_xticks([0, 120, 440])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    ax.xaxis.tick_bottom()
    ax.axhline(0, lw=1, c='k', ls='--')
    ax.scatter([121], [60], marker='v', c='k', s=40)

    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'task_plot'), transparent=True)


def select_epoch(df, n_blocks=15, day=1, epoch='early'):
    """Extract trials of task epoch

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level, trial-wise data
    n_blocks : int, optional
        Number of blocks to include, by default 15 (length of gradients window)
    day : int, optional
        Day/session, by default 1
    epoch : str, optional
        Task epoch to use, by default 'early'

    Returns
    -------
    pd.DataFrame
        Subject-level median error data
    """
    q = f"Day == 'Day {day}' & Block == 'Rotation'"
    if epoch == 'early': 
        q +=" & trial_block <= @n_blocks"
    elif epoch == 'late':
        max_block = df['trial_block'].max()
        init_block = max_block - n_blocks
        q +=" & trial_block > @init_block"
    elif epoch == 'washout':
        q = f"Day == 'Day {day}' & Block == 'Washout'"
    elif epoch == 'base':
        q = f"Day == 'Day {day}' & Block == 'Baseline'"

    return df.query(q)


def plot_epoch_error(df, name='early'):
    """Plot binned error curves for a given task epoch

    Parameters
    ----------
    data : pd.DataFrame
        Original trial-by-trial data
    name : str, optional
        Task epoch, by default 'early'
    """
    binned = df.groupby(['Number', 'trial_block']).mean().reset_index()
    medians = df.groupby('Number')['Error_deg'] \
                     .agg('median') \
                     .reset_index() \
                     .rename(columns={'Error_deg': 'median_error'})
    binned = binned.merge(medians, on='Number')

    fig, ax = plt.subplots(figsize=(2.5, 3))
    sns.lineplot(x='trial_block', y='Error_deg', data=binned, estimator=None,
                 units='Number', hue='median_error', lw=.5, ax=ax, 
                 palette=sns.cubehelix_palette(as_cmap=True))
    sns.lineplot(x='trial_block', y='Error_deg', data=binned, color='k', 
                 lw=1.5, ax=ax, ci=None)
    ax.legend_.remove()

    if name == 'early':
        xticks = np.arange(0, 20, 5)
    elif name == 'late':
        xticks = np.arange(25, 45, 5)
    else:
        xticks = None

    ax.set(xlabel='Trial block', ylabel='Angular error (°)', 
           xticks=xticks, yticks=np.arange(-30, 90, 15), 
           ylim=(-30, 75))
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(
        Config().figures, 'behaviour', f'{name}_learning_plot.png')
    )


def compute_error(df):
    """Compute median error

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level, trial-wise data of a specific epoch

    Returns
    -------
    pd.DataFrame
        Subject-level median error data
    """
    return df.groupby("Number")['Error_deg'] \
             .agg('median') \
             .reset_index()


def plot_error_distribution(df, name='early', ymax=45):
    """Show sample distribution of median epoch error

    Parameters
    ----------
    df : pd.DataFrame
        Median error data
    name : str, optional
        Task epoch, by default 'early'
    ymax : int, optional
        Max y-limit in plot, by default 45
    """
    plot_df = df.copy()
    plot_df['x'] = 1
    fig, ax = plt.subplots(figsize=(1.5, 2.8))

    box_line_color = 'k'
    sns.boxplot(x='x', y='Error_deg', data=plot_df, color='silver', 
                boxprops=dict(edgecolor=box_line_color), 
                medianprops=dict(color=box_line_color),
                whiskerprops=dict(color=box_line_color),
                capprops=dict(color=box_line_color),
                showfliers=False, width=.5)
    
    cmap = sns.cubehelix_palette(as_cmap=True)
    np.random.seed(1)
    jitter = np.random.uniform(.01, .4, len(plot_df['x']))
    ax.scatter(x=plot_df['x'] + jitter , y=plot_df['Error_deg'], 
               c=plot_df['Error_deg'], ec='k', linewidths=1, cmap=cmap, 
               clip_on=False)
    ax.set(xlabel=' ', ylabel='Median angular error (°)', xticks=[], 
           yticks=np.arange(-15, ymax + 15, 15))
    sns.despine(bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(
        Config().figures, 'behaviour', f'{name}_error_distribution.png')
    )


def correlation_map(error, gradients, name, sig_style=None, 
                    lateral_only=False):
    """Compute correlation between error and region eccentricity and display
    on surface

    Parameters
    ----------
    error : pd.DataFrame
        Subject-level median error data
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    name : str
        Name/label for analysis, as part of output filename

    Returns
    -------
    pd.DataFrame
        Region-wise correlation results
    """
    data = gradients[['sub', 'roi', 'roi_ix', 'distance']] \
                .pivot(index='sub', columns='roi', values='distance')
    # preserve original ROI order
    data = data[gradients['roi'].unique().tolist()]

    # double check!
    assert np.array_equal(error['Number'], data.index.values)

    res = data.apply(lambda x: pearsonr(error['Error_deg'], x), 
                     axis=0)
    res = res.T.rename(columns={0: 'r', 1: 'p'})
    _, res['p_fdr'] = pg.multicomp(res['p'].values, method='fdr_bh')

    config = Config()
    rvals = plotting.weights_to_vertices(res['r'], config.atlas)
    pvals = plotting.weights_to_vertices(res['p'], config.atlas)
    pvals = np.where(pvals < .05, 1, 0)
    qvals = plotting.weights_to_vertices(res['p_fdr'], config.atlas)
    qvals = np.where(qvals < .05, 1, 0)
    
    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    
    vmax = np.max(abs(res['r']))    
    cmap = 'RdBu_r'

    if lateral_only:
        p1 = Plot(surfaces['lh'], surfaces['rh'], views='lateral', 
                  layout='column', size=(250, 350), zoom=1.5)   
    else:
        p1 = Plot(surfaces['lh'], surfaces['rh'])
    
    p2 = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', 
                size=(150, 200), zoom=3.3)
    for p, suffix in zip([p1, p2], ['', '_dorsal']):
        p.add_layer(**sulc_params)
        
        cbar = True if suffix == '_dorsal' else False
        if sig_style is None:
            p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
            p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                        cbar=False, as_outline=True, cmap='viridis')
        elif sig_style == 'uncorrected':
            x = rvals * pvals
            vmin = np.nanmin(x[np.abs(x) > 0])
            cmap = cmr.get_sub_cmap(cmap, vmin/vmax, 1)
            p.add_layer(x, cmap=cmap, color_range=(vmin, vmax))
            p.add_layer((rvals*qvals != 0).astype(float), cbar=False, 
                        as_outline=True, cmap='gnuplot')
        elif sig_style == 'corrected':
            x = rvals * qvals
            vmin = np.nanmin(x[np.abs(x) > 0])
            p.add_layer(x, cbar=cbar, cmap=cmr.get_sub_cmap(cmap, .66, 1),  
                        color_range=(vmin, vmax))
            p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                        cbar=False, as_outline=True, cmap='binary')
        
        if suffix == '_dorsal':
            cbar_kws = dict(location='bottom', decimals=2, fontsize=10, 
                            n_ticks=2, shrink=.4, aspect=4, draw_border=False, 
                            pad=.05)
            fig = p.build(cbar_kws=cbar_kws)
        else:
            fig = p.build()
    
        if sig_style is None:
            suffix = suffix + '_uthr'
        prefix = os.path.join(FIG_DIR, f'{name}_correlation_map{suffix}')
        fig.savefig(prefix, dpi=300)

    return res


def plot_region_correlations(gradients, error):
    """Plot scatterplot for exemplar regions

    Parameters
    ----------
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    error : _type_
        Subject-level median error data
    """
    # pre-determined regions
    rois = ['7Networks_LH_SomMot_63', '7Networks_LH_SomMot_45']
    df = gradients.query("roi in @rois")
    df = df.merge(error, left_on='sub', right_on='Number')

    g = sns.lmplot(x='distance', y='Error_deg', col='roi', data=df,
                   scatter_kws={'color': 'k', 'clip_on': False}, 
                   line_kws={'color': 'k'}, sharex=False, height=2.3, aspect=.8)
    g.set_xlabels('Eccentricity')
    g.set_ylabels('Median angular error (°)')
    g.set(ylim=(-15, 45), yticks=np.arange(-15, 60, 15))
    g.set_titles('')
    g.tight_layout()
    
    g.savefig(os.path.join(FIG_DIR, 'example_correlations'), 
              transparent=True)


def network_correlation_analysis(gradients, error, label_table, networks=17):
    """Perform network-level correlation eccentricity-behaviour correlation

    Parameters
    ----------
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    error : pd.DataFrame
        Subject-level median error dataset
    label_table : pd.DataFrame
        Label file to map between 7 and 17 network Yeo versions
    networks : int, optional
        Number of Yeo networks, either 7 or 17, by default 17

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Correlation results and network-level eccentricity
    """
    data = gradients.copy()
    data = data.merge(label_table, left_on='roi', right_on='name_7networks', 
                      how='left')
    data = parse_roi_names(data, f'name_{networks}networks')

    network_ecc = data.groupby(['sub', 'network'])['distance'] \
                      .mean() \
                      .reset_index()
    data = network_ecc.merge(error, left_on='sub', 
                             right_on='Number', how='left')
    
    res =  data.groupby('network') \
               .apply(lambda x: pg.corr(x['distance'], x['Error_deg']))

    return fdr_correct(res, 'p-val'), data


def networks_plot(data):
    """Network correlation bar plot

    Parameters
    ----------
    data : pd.DataFrame
        Network-level correlation data

    Returns
    -------
    matplotlib.figure.Figure
        Bar plot
    """
    df = data.copy()
    df = df.sort_values('r', ascending=False).reset_index()

    # align colourmap to order of networks
    cmap = plotting.yeo_cmap(networks=17)
    colors = [cmap[i] for i in df['network']]

    fig, ax = plt.subplots(figsize=(3, 3))
    ticks = np.arange(df.shape[0])
    ax.bar(ticks, df['r'], color=colors)
    ax.set_xticks(ticks)
    ax.set_xticklabels(df['network'], rotation=90, ha="center")
    ax.tick_params(axis="x", bottom=False)

    ax.axhline(0, ls='-', lw=1, c='k', zorder=-1)
    ax.set(ylabel='Correlation ($\it{r}$)')

    sns.despine(bottom=True)
    fig.tight_layout()
    return fig


def network_scatter(data, network, xlim=None):
    """Plot network-level correlation scatterplot 

    Parameters
    ----------
    data : pd.DataFrame
        Network correlational data
    network : str
        Network name
    xlim : tuple of float, optional
        x axis limits, by default None, which will automatically determine 
        appropriate limits

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot
    """
    df = data.query("network == @network")

    color = plotting.yeo_cmap(networks=17)[network] 
    fig, ax = plt.subplots(figsize=(2.1, 2.1))
    g = sns.regplot(x='distance', y='Error_deg', data=df, color=color, ax=ax, 
                    scatter_kws=dict(clip_on=False, alpha=1))
    
    ax.set_xlabel('Network eccentricity')
    ax.set_ylabel('Median angular error (°)')

    if xlim is None:
        xlim = (np.round(df['distance'].min(), 1) - .1, None)
    ax.set(ylim=(-15, 45), yticks=np.arange(-15, 60, 15), xlim=xlim)
    
    sns.despine()
    fig.tight_layout()
    
    return fig


def main():
    config = Config()

    df = pd.read_csv(os.path.join(config.data_dir, 'subject_behavior.csv'))
    df['Error_deg'] = np.degrees(df['Error'])

    df, _ = bin_data(df)

    # methods learning fig
    task_behaviour_plot(df)

    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )
    early_gradients = gradients.query("epoch == 'early'")
    for epoch in ['early', 'late']:

        epoch_behav = select_epoch(df, epoch=epoch)
        plot_epoch_error(epoch_behav, epoch)
        error = compute_error(epoch_behav)
        plot_error_distribution(error)
        
        # whole brain correlation map w/ significance
        res = correlation_map(error, early_gradients, epoch, 
                            sig_style=None)
        res.reset_index().to_csv(
            os.path.join(config.results, f'{epoch}_correlations.tsv'), 
            index=False, sep='\t'
        )

        schaefer_labels = pd.read_csv(
            os.path.join(
                config.resources, 
                'atlases', 
                'Schaefer2018_1000Parcels_labels.csv'
            )
        )
        network_stats, network_data = network_correlation_analysis(
            early_gradients, error, schaefer_labels, networks=17
        )
        network_stats.reset_index().to_csv(
            os.path.join(config.results, f'{epoch}_network_correlations.tsv'), 
            index=False, sep='\t'
        )

        fig = networks_plot(network_stats)
        prefix = os.path.join(FIG_DIR, f'17_network_correlations_{epoch}.png')
        fig.savefig(prefix, dpi=300)

        if epoch == 'early':
            # figure 5 example scatterplots
            plot_region_correlations(early_gradients, error)
            for i in ['DorsAttnB', 'ContA']:
                fig = network_scatter(network_data, i)
                prefix = os.path.join(FIG_DIR, f'{i}_scatter.png')
                fig.savefig(prefix, dpi=300)

    
if __name__ == '__main__':
    main()


