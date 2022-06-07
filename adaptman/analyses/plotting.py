
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn.plotting import cm
import cmasher as cmr
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall

from adaptman.config import Config
from adaptman.utils import get_surfaces

def set_plotting():
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams["savefig.format"] = 'png'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = 'tight'
    sns.set_context('paper')


def save_prefix(prefix):
    """Append figure directory to file name prefix

    Parameters
    ----------
    prefix : str
        File name without file extension

    Returns
    -------
    str
        Complete path to file name prefix
    """
    config = Config()
    return os.path.join(config.figures, prefix + '_')


def _align_labels_to_atlas(x, source_labels, target_labels):
    """Match labels to corresponding vertex labels"""

    target = np.unique(target_labels)[1:]
    df1 = pd.DataFrame(target, index=target)
    df2 = pd.DataFrame(x, index=source_labels)
    return pd.concat([df1, df2], axis=1).iloc[:, 1:].values


def weights_to_vertices(data, target, labels=None):
    """Map weights (e.g., gradient loadings) to vertices on surface

    If `labels` is not specifiied, values in `data` are mapped to `target` in 
    ascending order.

    Parameters
    ----------
    data : numpy.ndarray or str
        Array containing region weights of shape (n_regions, n_features). If
        more than one feature/column is detected, then brain maps for each 
        feature are created. If a string, must be a valid CIFTI file name
    target : str
        CIFTI file name (dlabel or dscalar) that defines vertex-space for 
        mapping 
    labels : numpy.ndarray
        Numeric labels for each region (i.e. row of `data`) as they appear in 
        the atlas vertices. Required when `data` contains fewer regions than
        total regions in `target`, as is the case when `data` is a result of 
        some thresholded/indexing (e.g., `data` only contains weights of 
        significant regions). By default None.

    Returns
    -------
    numpy.ndarray
        Array of mapped vertices
    """
    if isinstance(target, str): 
        vertices = nib.load(target).get_fdata().ravel()
    else:
        vertices = target.ravel()

    if labels is not None:
        data = _align_labels_to_atlas(data, labels, vertices)

    mask = vertices != 0
    map_args = dict(target_lab=vertices, mask=mask, fill=np.nan)
    if (len(data.shape) == 1) or (data.shape[1] == 1):
        weights = map_to_labels(data.ravel(),  **map_args)
    else:
        weights = [map_to_labels(x, **map_args) for x in data.T]
    return weights


def get_sulc():
    """Get sulcal depth map for plotting style"""
    config = Config()
    surf_path = os.path.join(config.resources, 'surfaces')
    img = os.path.join(surf_path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')
    vertices = nib.load(img).get_fdata().ravel()
    return add_fslr_medial_wall(vertices)


def yeo_cmap(as_palette=False, networks=7):
    """Color map for Yeo 7 network parcellation

    Parameters
    ----------
    as_palette : bool, optional
        Return as seaborn.palettes._ColorPalette instead of dictionary. By
        default False

    Returns
    -------
    dict or seaborn.palettes._ColorPalette
        Dictionary of RBG values associated with each network, or a seaborn 
        color palette in the same order
    """
    if networks == 17:
        cmap = {
            'VisCent': (120, 18, 136),
            'VisPeri': (255, 0, 2),
            'SomMotA': (70, 130, 181),
            'SomMotB': (43, 204, 165),
            'DorsAttnA': (74, 156, 61),
            'DorsAttnB': (0, 118, 17),
            'SalVentAttnA': (196, 58, 251),
            'SalVentAttnB': (255, 153, 214),
            'TempPar': (9, 41, 250),
            'ContA': (230, 148, 36),
            'ContB': (136, 50, 75),
            'ContC': (119, 140, 179),
            'DefaultA': (255, 254, 1),
            'DefaultB': (205, 62, 81),
            'DefaultC': (0, 0, 132),
            'LimbicA': (224, 248, 166),
            'LimbicB': (126, 135, 55)
        }
    else:
        cmap = {
            'Vis': (119, 20, 140),
            'SomMot': (70, 126, 175), 
            'DorsAttn': (0, 117, 7), 
            'SalVentAttn': (195, 59, 255), 
            'Limbic': (219, 249, 165), 
            'Cont': (230, 149, 33), 
            'Default': (205, 65, 80) 
        }
    cmap = {k: np.array(v) / 255 for k, v in cmap.items()}
    if as_palette:
        return sns.color_palette(cmap.values())
    else:
        return cmap


def plot_cbar(cmap, vmin, vmax, orientation='vertical', size=None, n_ticks=2, 
              decimals=2, fontsize=12, show_outline=False, as_int=False):
    """Plot standalone colorbar

    Parameters
    ----------
    cmap : str or matplotlib.colors.ListedColormap
        Color map. Can either be specified as a named matplotlib color map, or 
        an instance of ListedColormap 
    vmin : float
        Minimum value of colorbar
    vmax : float
        Maximum value of colorbar
    orientation : {'vertical', 'horizontal'}, optional
        Orientation of colorbar, by default 'vertical'
    size : tuple, optional
        Custom figure size, by default None
    n_ticks : int, optional
        Number of ticks to show on colour bar, by default 2 (only the min 
        and max)
    decimals : int, optional
        Number of decimals to show, by default 2
    """
    # default sizing
    if size is None and (orientation == 'vertical'):
        size = (.3, 4)
    if size is None and (orientation == 'horizontal'):
        size = (4, .3)

    x = np.array([[0,1]])
    plt.figure(figsize=size)
    img = plt.imshow(x, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])

    # configure scale
    ticks = np.linspace(0, 1, n_ticks)
    tick_labels = np.around(np.linspace(vmin, vmax, n_ticks), decimals)
    cbar = plt.colorbar(orientation=orientation, cax=cax, ticks=ticks)
    if as_int:
        cbar.set_ticklabels(tick_labels.astype(int))
    else:
        cbar.set_ticklabels(tick_labels)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if not show_outline:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
    return cbar


def plot_3d(x, y, z, color=None, ax=None, view_3d=(35, -110), **kwargs):
    """Plot 3D scatter plot of region loadings/weights

    Parameters
    ----------
    x, y, z : array-like
        Data to plot in X, Y, and Z dimensions, respectively
    color : array-like, optional
        Colours to assign each element in data. Must be same length as x, y, 
        and z. By default None
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Existing matplotlib axis. If None, a new Figure and axis will be 
        created. By default None
    view_3d : tuple, optional
        Sets initial view as (elevation, azimuth). By default None
    **kwargs : dict, optional
        Other keyword arguments for `matplotlib.pyplot.scatter`

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        3D scatter plot
    """
    if ax is None:
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs=x, ys=y, zs=z, c=color, **kwargs)
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    if view_3d is not None:
        ax.view_init(elev=view_3d[0], azim=view_3d[1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    return ax


def stat_cmap():
    return cmr.get_sub_cmap(cm.cyan_orange, .05, .95)


def pairwise_stat_maps(data, prefix, dorsal=True, posterior=True, vmax='auto', 
                       vmin='auto', cbar_orientation='vertical'):
    """Plot pairwise comparisons t-maps on brain surfaces

    Parameters
    ----------
    data : pandas.DataFrame
        Pairwise t-test results
    prefix : str
        File name prefix
    dorsal : bool, optional
        Plot dorsal view, by default True
    posterior : bool, optional
        Plot posterior view, by default True
    vmax : float or str, optional
        Max value in colour range, by default 'auto'
    vmin : float or str, optional
        Min value in colour range, by default 'auto'
    cbar_orientation : str, optional
        Colour bar orientation, either 'vertical' or 'horizontal', 
        by default 'vertical'

    Returns
    -------
    matplotlib.figure.Figure
        Stat maps
    """
    pairwise_contrasts = data.groupby(['A', 'B'])
    list_ = []
    for name, g in pairwise_contrasts:
        contrast = f'{name[0]}_{name[1]}'
        # get stats of only significant regions
        sig = g[g['sig_corrected'].astype(bool)]
        sig.set_index('roi_ix', inplace=True)
        sig = sig[['T']]
        sig = sig.rename(columns={'T': contrast + '_T'})
        list_.append(sig)
    df = pd.concat(list_, axis=1)

    # flip sign so that the second condition is the positive condition 
    # (because epochs are both alphabetical AND chronological, the 
    # positive condition is always the subsequent epoch)
    tvals = -df.filter(like='T')
    
    if vmax == 'auto':
        vmax = np.nanmax(tvals.values)
        
    if vmin == 'auto':
        vmin = np.nanmin(np.abs(tvals.values))

    # draw separate pos/neg cmaps
    if cbar_orientation == 'vertical':
        size = (.2, 1)
    else:
        size = (.8, .25)

    cmap = stat_cmap()
    pos_cmap = cmr.get_sub_cmap(cmap, .51, 1)
    pos_cmap = cmr.get_sub_cmap(pos_cmap, vmin/vmax, 1)
    plot_cbar(pos_cmap, vmin, vmax, cbar_orientation, size=size, n_ticks=2)
    plt.savefig(prefix + 'cbar_pos')

    neg_cmap = cmr.get_sub_cmap(cmap, 0, .5)
    neg_cmap = cmr.get_sub_cmap(neg_cmap, 0, 1 - vmin/vmax)
    plot_cbar(neg_cmap, -vmax, -vmin, cbar_orientation, size=size, 
              n_ticks=2)
    plt.savefig(prefix + 'cbar_neg')

    config = Config()
    surfaces = get_surfaces()
    sulc = get_sulc()
    # cmap = cmr.get_sub_cmap(cm.cyan_orange, .15, .83)
    cmap = stat_cmap()
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(-vmax, vmax))
    outline_params = dict(cbar=False, cmap='binary', as_outline=True)

    for i in tvals.columns:
        contrast = i[:-2].replace('_', '_vs_')
        x = weights_to_vertices(tvals[i], config.atlas, 
                                           tvals.index.values)

        # row 
        p = Plot(surfaces['lh'], surfaces['rh'], layout='row', 
                 mirror_views=True, size=(800, 200), zoom=1.2)
        p.add_layer(**sulc_params)
        p.add_layer(x, **layer_params)
        p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
        fig = p.build(colorbar=False)
        fig.savefig(prefix + contrast)

        if dorsal:
            p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', 
                    size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, **layer_params)
            p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build(colorbar=False)
            fig.savefig(prefix + contrast + '_dorsal')

        if posterior:
            p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', 
                    size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, **layer_params)
            p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build(colorbar=False)
            fig.savefig(prefix + contrast + '_posterior')

    return fig


def plot_timeseries(fname):
    """Plot exemplar timeseries for methods figure

    Parameters
    ----------
    fname : str
        Timeseries file

    Returns
    -------
    matplotlib.figure.Figure
        Timeseries figure
    """
    df = pd.read_table(fname)
    df = df.iloc[:, [49, 144, 480]]
    cmap = yeo_cmap()

    scale_factor = 7
    fig, ax = plt.subplots(figsize=(5, 1))
    for i, roi in enumerate(['Default', 'SomMot', 'Vis']):
        
        if i == 0:
            const = i * scale_factor
        else:
            const = (i + 1) * scale_factor
        ax.plot(df.iloc[:, i] + const, c=cmap[roi], lw=1)
    
    ax.set_axis_off()
    fig.tight_layout()
    return fig
    