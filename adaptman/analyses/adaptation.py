"""Main analysis of eccentricity changes during adaptation

Notes
-----
- Eccentricity is precomputed in the eccentricity.py module
- Washout is separately analyzed in washout.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.cluster import KMeans
import cmasher as cmr
from surfplot import Plot

from adaptman.config import Config
from adaptman.utils import parse_roi_names, get_surfaces, test_regions
from adaptman.analyses import plotting

plotting.set_plotting()


def eccentricity_analysis(data):
    """Determine if regions show significant changes in eccentricity across
    task epochs

    Basic mass-univariate approach that performs an F-test across each region,
    followed by follow-up paired t-tests on significant regions

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        ANOVA and post-hoc stats tables, respectively
    """
    anova = test_regions(data)

    # post hoc analysis
    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi'].tolist()
    if sig_regions:
        post_data = data[data['roi'].isin(sig_regions)]
        posthoc = test_regions(post_data, 'ttest')  
    else:
        # no significant anova results
        posthoc = None

    return anova, posthoc


def mean_stat_map(data, out_dir, centering='baseline'):
    """Plot the average eccentricity for each region on brain.

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    out_dir : str
        Figure save/output directory
    centering : str, optional
        Centering approach, if applicable. Use 'baseline' to subtract 
        Baseline eccentricity and show relative changes during adaptation, 
        'mean' for mean-centering across all epochs, and None to show original 
        eccentricity values

    Returns
    -------
    matplotlib.figure.Figure
        Mean stat map
    """
    mean_ecc = data.groupby(['epoch', 'roi', 'roi_ix'])['distance'] \
                   .mean() \
                   .reset_index() \
                   .sort_values(['epoch', 'roi_ix'])
    
    # roi by epoch (in chronological order) data for plotting
    epoch_ecc = pd.DataFrame({name: g['distance'].values 
                              for name, g in mean_ecc.groupby('epoch')})
    
    if centering is not None:
        if centering == 'mean':
            epoch_ecc = epoch_ecc.apply(lambda x: x - x.mean(), axis=1)
            prefix = 'centered_mean_ecc_'
        elif centering == 'baseline':
            epoch_ecc['early'] -= epoch_ecc['base']
            epoch_ecc['late'] -= epoch_ecc['base']
            epoch_ecc.drop('base', axis=1, inplace=True)
            prefix = 'relative_mean_ecc_'

        cmap = sns.diverging_palette(210, 5, s=100, l=30, sep=10, 
                                    as_cmap=True)
        vmax = np.around(np.nanmax(epoch_ecc), 2)
        vmin = -vmax
        n_ticks = 3
    else:
        cmap = 'viridis'
        vmax = np.nanmax(epoch_ecc)
        vmin = np.nanmin(epoch_ecc)
        n_ticks = 2
        prefix = 'mean_ecc_'

    prefix = os.path.join(out_dir, prefix)
    plotting.plot_cbar(cmap, vmin, vmax, 'vertical', size=(.2, 1), 
                         n_ticks=n_ticks)
    plt.savefig(prefix + 'cbar') 

    config = Config()
    surfaces = get_surfaces()

    for i in epoch_ecc.columns:
        x = plotting.weights_to_vertices(epoch_ecc[i], config.atlas)

        p = Plot(surfaces['lh'], surfaces['rh'], layout='row', 
                 mirror_views=True, size=(800, 200), zoom=1.2)
        p.add_layer(x, cmap=cmap, color_range=(vmin, vmax), cbar=False)
        fig = p.build()
        fig.savefig(prefix + i + '_brain')

    return fig


def plot_mean_scatters(data, out_dir, view_3d=(30, -110), eccentricity=False):
    """Plot scatter plot of mean gradients for each epoch

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    out_dir : str
        Figure save/output directory
    view_3d : tuple, optional
        Viewpoint as (evelation, rotation), by default (30, -110)
    eccentricity : bool, optional
        Whether plot region eccentricity via color scaling, by default False
    """
    k = [f'g{i}' for i in np.arange(3) + 1]
    mean_loadings = data.groupby(['epoch', 'roi'])[k + ['distance']] \
                        .mean() \
                        .reset_index()
    mean_loadings = parse_roi_names(mean_loadings)

    if eccentricity:
        c_col = 'distance'
        cmap = 'viridis'
        vmax = np.nanmax(mean_loadings['distance'])
        vmin = np.nanmin(mean_loadings['distance'])
        suffix = 'scatter_ecc'
    else:
        c_col='c'
        cmap = plotting.yeo_cmap()
        mean_loadings['c'] = mean_loadings['network'].apply(lambda x: cmap[x])
        vmax, vmin = None, None
        suffix = 'scatter'

    for epoch in ['base', 'early', 'late']:
        df = mean_loadings.query("epoch == @epoch")

        x, y, z = k
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax = plotting.plot_3d(df[x], df[y], df[z], color=df[c_col], 
                                s=10, lw=0, ax=ax, view_3d=view_3d, 
                                vmax=vmax, vmin=vmin)
        ax.set(xlim=(-3.5, 5), ylim=(-3.5, 5), zlim=(-3.5, 4))
        
        if epoch == 'base':
            title = 'Baseline'
        else:
            title = epoch
        ax.set_title(title.title())
        
        prefix = os.path.join(out_dir, f'mean_{epoch}_')
        fig.savefig(prefix + suffix)

    
def anova_stat_map(anova, out_dir, outline=True):
    """Plot thresholded mass-univariate ANOVA results 

    Threshold set as q < .05, where q = FDR-corrected two-tailed p values

    Parameters
    ----------
    anova : pandas.DataFrame
        ANOVA results

    Returns
    -------
    pandas.DataFrame
        ANOVA results
    """
    df = anova.query("sig_corrected == 1")
    fvals = df['F'].values
    vmax = int(np.nanmax(fvals)) 
    vmin = np.nanmin(fvals)

    # get orange (positive) portion. Max reduced because white tends to wash 
    # out on brain surfaces
    cmap = cmr.get_sub_cmap(plotting.stat_cmap(), .5, 1)
    # get cmap that spans from stat threshold to max rather than whole range, 
    # which matches scaling of t-test maps
    cmap_min = vmin / vmax
    cmap = cmr.get_sub_cmap(cmap, cmap_min, 1)

    plotting.plot_cbar(cmap, vmin, vmax, 'horizontal', size=(1, .3), 
                         n_ticks=2)
    prefix = os.path.join(out_dir, 'anova')
    plt.savefig(prefix + '_cbar')

    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    x = plotting.weights_to_vertices(fvals, Config().atlas, 
                                       df['roi_ix'].values)
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(vmin, vmax))
    outline_params = dict(data=(np.abs(x) > 0).astype(bool), cmap='binary', 
                          cbar=False, as_outline=True)

    # 2x2 grid
    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)

    cbar_kws = dict(n_ticks=2, aspect=8, shrink=.15, draw_border=False)
    fig = p.build(cbar_kws=cbar_kws)
    fig.savefig(prefix)

    # dorsal views
    p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), 
             zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_dorsal')

    # posterior views
    p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', 
             size=(150, 200), zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_posterior')

    return x


def _ensemble_cmap(as_cmap=False):

    colors = ['tab:cyan', 'orangered', 'tab:purple', 'tab:olive']
    if as_cmap:
        return LinearSegmentedColormap.from_list('cmap', colors, N=4)
    else:
        return dict(zip(range(1, 5), colors))


def ensemble_analysis(gradients, anova, out_dir, k=3):
    """Cluster significant regions into functional ensembles

    Parameters
    ----------
    gradients : pandas.DataFrame
        Subject gradient data
    anova : pandas.DataFrame
        Pre-computed ANOVA data
    out_dir : str
        Figure save/output directory
    k : int, optional
        Number of gradients/dimensions to include, by default 3

    Returns
    -------
    pandas.DataFrame
        Region ensemble assignments
    """
    cols = [f'g{i}' for i in np.arange(k) + 1]
    base_loadings = gradients.query("epoch == 'base'")

    sig_rois = anova.query("sig_corrected == 1")['roi'].tolist()
    df = base_loadings.query("roi in @sig_rois") \
                      .groupby(['epoch', 'roi'], sort=False)[cols] \
                      .mean() \
                      .reset_index()
    kmeans = KMeans(n_clusters=4, random_state=1234)
    kmeans.fit(df[cols])
    df['ensemble'] = kmeans.labels_ + 1

    res = base_loadings.merge(df[['roi', 'ensemble']], on='roi', how='left')

    # brain plot
    x = plotting.weights_to_vertices(res['ensemble'].values, Config().atlas)
    x = np.nan_to_num(x)
    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    cmap = _ensemble_cmap(True)

    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    p.add_layer(x, cbar=None, cmap=cmap)
    p.add_layer((np.abs(x) > 0).astype(bool), as_outline=True, 
                cbar=None, cmap='binary')
    fig = p.build()
    fig.savefig(os.path.join(out_dir, 'anova_ensembles.png'), dpi=300)

    data = gradients.merge(df[['roi', 'ensemble']], on='roi', how='left')
    data = data.groupby(['sub', 'epoch', 'ensemble'])['distance'] \
               .mean() \
               .reset_index()

    colors = list(_ensemble_cmap().values())
    g = sns.FacetGrid(data=data, col_wrap=2, col='ensemble', hue='ensemble',
                      palette=colors, height=1.8)
    g.map_dataframe(sns.lineplot, x='epoch', y='distance', ci=None, 
                    marker='o', ms=5, lw=1.2, mfc='w', mec='k', color='k')
    g.map_dataframe(sns.stripplot, x='epoch', y='distance', jitter=.1, 
                    zorder=-1, s=4, alpha=.752)
    g.set_axis_labels('', "Eccentricity")
    g.set(xticklabels=['Baseline', 'Early', 'Late'])
    g.set_titles('')
    g.savefig(os.path.join(out_dir, 'ensemble_ecc_plot.png'), dpi=300)
    return df[['roi', 'ensemble']]


def plot_displacements(data, anova, k=3, ax=None, hue='network'):
    """Plot low-dimensional displacements of regions that show significant 
    ANOVA results (i.e. changes in eccentricity)

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    anova : pandas.DataFrame
        ANOVA results
    k : int, optional
        Number of gradients to include, by default 3
    ax : matplotlib.axes._axes.Axes, optional
        Preexisting matplotlib axis, by default None

    Returns
    -------
    matplotlib.figure.Figure and/or matplotlib.axes._axes.Axes
        Displacement scatterplot figure
    """
    if isinstance(k, int):
        k = [f'g{i}' for i in np.arange(k) + 1]

    ensb = data[['roi', 'ensemble']].groupby('roi', sort=False).first()

    mean_loadings = data.groupby(['epoch', 'roi'])[k].mean().reset_index()
    mean_loadings = parse_roi_names(mean_loadings)
    mean_loadings = mean_loadings.merge(ensb, on='roi', how='left')

    base = mean_loadings.query("epoch == 'base'")
    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi']
    sig_base = base[base['roi'].isin(sig_regions)]
    shifts = mean_loadings[mean_loadings['roi'].isin(sig_regions)]

    if hue == 'network':
        cmap = plotting.yeo_cmap()
    elif hue == 'ensemble':
        cmap = _ensemble_cmap()
    if len(k) == 2:
        x, y = k
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        # all regions  
        sns.scatterplot(x=x, y=y, data=base, color='k', alpha=.3, s=5, 
                        linewidths=0, legend=False, ax=ax)

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            val = roi_df[hue].iloc[0]
            ax.plot(xx, yy, lw=1, c=cmap[val])
            
            arrowprops = dict(lw=.1, width=.1, headwidth=4, headlength=3, 
                              color=cmap[val])
            ax.annotate(text='', xy=(xx[-1], yy[-1]), xytext=(xx[-2], yy[-2]), 
                        arrowprops=arrowprops)
        
        # plot color-coded markers of significant regions
        sns.scatterplot(x=x, y=y, data=sig_base, hue=hue, s=16, 
                        edgecolor='k', palette=cmap, linewidths=1, ax=ax, 
                        legend=False, zorder=20)
        sns.despine()
        return ax
    
    elif len(k) == 3:
        x, y, z = k
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(nrows=10, ncols=10)
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')

        # remove sig regions so that their points don't obstruct their 
        # colour-coded points plotted below
        base_nonsig = base[~base['roi'].isin(sig_regions)]
        ax1 = plotting.plot_3d(base_nonsig[x], base_nonsig[y], base_nonsig[z],
                                color='gray', alpha=.3, s=1, ax=ax1, 
                                view_3d=(35, -110))
        ax1.set(xticks=range(-4, 6))

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            zz = roi_df[z].values
            val = roi_df[hue].iloc[0]
            ax1.plot(xs=xx, ys=yy, zs=zz, lw=1, c=cmap[val])
        
        # color-coded significant regions
        sig_base['c'] = sig_base[hue].apply(lambda x: cmap[x])
        ax1 = plotting.plot_3d(sig_base[x], sig_base[y], sig_base[z], 
                                color=sig_base['c'], alpha=1, s=20,
                                ax=ax1, zorder=20, edgecolors='k', 
                                linewidths=.5)

        ax2 = fig.add_subplot(gs[:5, 6:9])
        ax2 = plot_displacements(data, anova, ['g1', 'g2'], ax=ax2)
        ax2.set(ylim=(-4, 5), xlim=(-4, 5), xticklabels=[], 
                xlabel='', ylabel='PC2')
        ax3 = fig.add_subplot(gs[5:, 6:9])
        ax3 = plot_displacements(data, anova, ['g1', 'g3'], ax=ax3)
        ax3.set(ylim=(-4, 5), xlim=(-4, 5), xticks=np.arange(-4, 6, 2), 
                xlabel='PC1', ylabel='PC3')

        fig.tight_layout()
        return fig, ax
    else:
        return None, None
    

def main():

    config = Config()
    fig_dir = os.path.join(Config().figures, 'adaptation')
    os.makedirs(fig_dir, exist_ok=True)

    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )
    gradients = gradients.query("epoch != 'washout'")
    anova_stats, ttest_stats = eccentricity_analysis(gradients)

    anova_stats.to_csv(os.path.join(config.results, 'ecc_anova_stats.tsv'), 
                       sep='\t', index=False)
    ttest_stats.to_csv(os.path.join(config.results, 'ecc_ttest_stats.tsv'), 
                       sep='\t', index=False)


    mean_stat_map(gradients, fig_dir)
    mean_stat_map(gradients, fig_dir, 'mean')
    mean_stat_map(gradients, fig_dir, 'baseline')
    
    anova_vertices = anova_stat_map(anova_stats, fig_dir)
    np.savetxt(os.path.join(config.results, 'anova_vertices.tsv'),
               anova_vertices)
    plotting.pairwise_stat_maps(ttest_stats, 
                                os.path.join(fig_dir, 'ecc_ttests_'))
    
    ensb = ensemble_analysis(gradients, anova_stats, fig_dir, config.k)
    gradients = gradients.merge(ensb, on='roi', how='left')
    anova_stats = anova_stats.merge(ensb, on='roi', how='left')

    # 3D plots
    if config.k == 3:
        plot_mean_scatters(gradients, fig_dir)
        plot_mean_scatters(gradients, fig_dir, eccentricity=True)
        fig, _ = plot_displacements(gradients, anova_stats, config.k)
        fig.savefig(os.path.join(fig_dir, 'displacements'))

        fig, _ = plot_displacements(gradients, anova_stats, config.k, 
                                     hue='ensemble')
        fig.savefig(os.path.join(fig_dir, 'displacements_by_ensemble'))


if __name__ == '__main__':
    main()
