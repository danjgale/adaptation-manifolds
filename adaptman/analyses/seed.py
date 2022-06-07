import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import pingouin as pg
import cmasher as cmr
from surfplot import Plot

from adaptman.config import Config
from adaptman.utils import (get_files, schaefer1000_roi_ix, fdr_correct, 
                           parse_roi_names)
from adaptman.analyses.plotting import (get_sulc, get_surfaces, set_plotting,
                                       weights_to_vertices)


set_plotting()


def find_cluster_seed(df, method='middle'):
    """Identify cluster seeds using region with max or closest-to-mean T value
    in contrast

    Parameters
    ----------
    df : pandas.DataFrame
        Contrast data (e.g., Early vs Baseline)
    method : str, optional
        Selection approach, by default 'middle', which is the closest-to-mean
        region

    Returns
    -------
    str
        Seed region name
    """
    if method == 'max':
        return df.loc[df['T'].idxmax()]
    elif method == 'middle':
        return df.loc[np.abs(df['T'] - df['T'].mean()).idxmin()]


def connect_seed(cmats, seed_region):
    """Extract seed connectivity by isolating row in connectivity matrix

    Parameters
    ----------
    cmats : list
        Connectivity matrices
    seed_region : str
        Seed region name

    Returns
    -------
    pandas.DataFrame
        Region connectivity profiles across subjects
    """
    list_ = []
    for i in cmats:
        cmat = pd.read_table(i, index_col=0)
        # isolate row of seed region
        res = pd.DataFrame(cmat.loc[seed_region].reset_index().values, 
                           columns=['roi', 'r'])
        
        res['roi_ix'] = schaefer1000_roi_ix()

        fname = os.path.basename(i)
        res['sub'] = fname.split('_')[0]
        res['epoch'] = fname.split('_')[-1].split('.')[0]
        list_.append(res)

    connectivity = pd.concat(list_)
    connectivity['r'] = connectivity['r'].astype(np.float)
    return connectivity


def seed_analysis(contrasts, clust_num, cmats, epochs):
    """Perform seed connectivity contrast analysis

    Parameters
    ----------
    contrasts : pandas.DataFrame
        Eccentricity contrast results
    clust_num : int
        Eccentricity contrast cluster number
    cmats : List
        Connectivity matrices
    epochs : _type_
        Task epochs to compare connectivity, not necessarily the same task 
        epochs from the eccentricity contrast  

    Returns
    -------
    pandas.DataFrame, str, pandas.DataFrame
        Seed connectivity results (region and networks), and seed name
    """
    seed = find_cluster_seed(contrasts.query("cluster == @clust_num"))
    connectivity = connect_seed(cmats, seed['roi'])

    df = connectivity.query("epoch in @epochs")
    res = df.groupby(['roi', 'roi_ix'], sort=False) \
            .apply(pg.pairwise_ttests, dv='r', within='epoch', subject='sub') \
            .reset_index()
    # swap sign so that B condition is the positive condition
    res['T'] = -res['T']
    res['sig'] = (res['p-unc'] < .05).astype(float)

    # get network-level changes
    df = parse_roi_names(df)
    networks = df.groupby(['sub', 'network', 'epoch']) \
                 .mean() \
                 .reset_index()
    return fdr_correct(res), seed, networks


def plot_seed_map(data, seed_region, sig_style=None, use_fdr=True, 
                  seed_color='yellow', show_left_vis=False):
    """Generate seed connectivity contrast maps

    Parameters
    ----------
    data : pandas.DataFrame
        Region-wise seed connectivity results
    seed_region : str
        Seed region name
    sig_style : str, optional
        Significance indication, by default None
    use_fdr : bool, optional
        If showing significance, show FDR-corrected results, by default True
    seed_color : str, optional
        Seed region color, by default 'yellow'
    show_left_vis : bool, optional
        Show left visual cortex, which is necessary for the visual seed only, 
        by default False

    Returns
    -------
    matplotlib.figure.Figure
        Seed contrast stat map
    """
    if use_fdr:
        sig_regions = data.query("sig_corrected == 1")
    else:
        sig_regions = data.query("sig == 1")
    
    seed = (data['roi'] == seed_region).astype(float)
    
    x = weights_to_vertices(data['T'].astype(float).values, Config().atlas, 
                            data['roi_ix'].values)
    y = weights_to_vertices(np.ones(len(sig_regions)), Config().atlas, 
                            sig_regions['roi_ix'].values)
    z = weights_to_vertices(seed.values, Config().atlas, data['roi_ix'].values)
    seed_cmap = LinearSegmentedColormap.from_list(
        'regions', [seed_color, 'k'], N=2
    )

    surfs = get_surfaces()
    sulc = get_sulc()
    if show_left_vis:
        p = Plot(surfs['lh'], views='posterior')
        sulc, x, y, z = [i[:int(len(i) / 2)] for i in [sulc, x, y, z]]
        p.add_layer(data=sulc, cmap='gray', cbar=False)
    else:    
        p = Plot(surfs['lh'], surfs['rh'])
        p.add_layer(data=sulc, cmap='gray', cbar=False)

    vmax = np.nanmax(np.abs(x))
    cmap = cmr.get_sub_cmap('seismic', .25, .76)
    if sig_style == 'trace':
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        p.add_layer(np.nan_to_num(y), as_outline=True, 
                    cmap='binary', cbar=False)
    elif sig_style == 'threshold':
        p.add_layer(x*np.nan_to_num(y), cmap=cmap, color_range=(-vmax, vmax))
    elif sig_style is None:
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))

    p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
    p.add_layer(np.nan_to_num(z), as_outline=True, cmap='binary', cbar=False)
    
    cbar_kws = dict(location='bottom', decimals=1, fontsize=14, n_ticks=3, 
                    shrink=.2, aspect=6, draw_border=False, pad=-.06)
    fig = p.build(cbar_kws=cbar_kws)

    return fig


def main():
    config = Config()
    cmats = get_files(
        os.path.join(
            config.dataset_dir, 'connectivity', 
            'sub*/sub*ses-01*rotation*.tsv'
        )
    )
    ttest_data = pd.read_table(
        os.path.join(config.results, 'ecc_ttest_stats.tsv')
    )

    fig_dir = os.path.join(config.figures, 'seed')
    os.makedirs(fig_dir, exist_ok=True)

    network_data = []
    seeds = []
    clusters = [1, 2, 4, 1]
    names = ['PMC', 'SPC', 'PMd', 'Vis']
        
    for clust, name in zip(clusters, names):

        if name == 'Vis':
            df = ttest_data.query("A == 'early' and B == 'late'")
        else:
            df = ttest_data.query("A == 'base' and B == 'early'")
        
        epochs = [['base', 'early'], ['late', 'early'], ['late', 'base']]
        for e in epochs:
            suffix = f'{e[1]}_vs_{e[0]}'
    
            res, seed, networks = seed_analysis(df, clust, cmats, e)
            networks['cluster'] = name
            network_data.append(networks)

            fig = plot_seed_map(res, seed['roi'], sig_style=None)
            out = os.path.join(fig_dir, f'{name}_{suffix}.png')
            fig.savefig(out, dpi=300)

            if name == 'Vis':
                fig = plot_seed_map(res, seed['roi'], sig_style=None, 
                                    show_left_vis=True)
                out = os.path.join(fig_dir, f'{name}_{suffix}_post.png')
                fig.savefig(out, dpi=300)

        seeds.append(seed)
    print(seeds)


if __name__ == '__main__':
    main()
