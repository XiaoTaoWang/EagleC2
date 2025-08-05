import itertools, cooler
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}

matplotlib.rcParams.update(new_rc_params)

def load_sv_full(fil):

    SVs = []
    with open(fil, 'r') as source:
        source.readline()
        for line in source:
            c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2, ng = line.rstrip().split()
            p1, p2 = int(p1), int(p2)
            prob1, prob2, prob3, prob4, prob5, prob6 = float(prob1), float(prob2), float(prob3), float(prob4), float(prob5), float(prob6)
            res1, res2 = int(res1), int(res2)
            SVs.append((c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2, ng))
    
    return SVs

class intraChrom(object):
    
    def __init__(self, uri, chrom, start, end, correct='sweight', figsize=(2.1, 1.5),
        n_rows=4, track_partition=[4,0.7,0.3,0.8], space=0.01):

        self.clr = cooler.Cooler(uri)
        self.res = self.clr.binsize

        fig = plt.figure(figsize=figsize)
        self.fig = fig
        self.grid = GridSpec(n_rows, 1, figure=fig, left=0.1, right=0.9,
                    bottom=0.1, top=0.9, hspace=space, height_ratios=track_partition)
        self.track_count = 0

        self.chrom = chrom
        self.start = start
        self.end = end

        M = self.clr.matrix(balance=correct, sparse=False).fetch((chrom, start, end))
        M[np.isnan(M)] = 0
        self.matrix = M

        # define my colormap (traditional w --> r)
        self.cmap = LinearSegmentedColormap.from_list('interaction',
                ['#FFFFFF','#FFDFDF','#FF7575','#FF2626','#F70000'])
    
    def matrix_plot(self, colormap='traditional', vmin=None, vmax=None, log=False,
        cbr_width=0.03, cbr_height=0.18, cbr_fontsize=4, no_colorbar=False):

        h_ax = self.fig.add_subplot(self.grid[self.track_count])
        self.track_count += 1

        heatmap_pos = h_ax.get_position().bounds

        M = self.matrix
        n = M.shape[0]

        # Create the rotation matrix
        t = np.array([[1,0.5], [-1,0.5]])
        A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)

        if colormap=='traditional':
            cmap = self.cmap
        else:
            cmap = colormap
        
        # Plot the Heatmap ...
        x = A[:,1].reshape(n+1, n+1)
        y = A[:,0].reshape(n+1, n+1)
        y[y<0] = -y[y<0]

        if vmax is None:
            vmax = np.percentile(M[M.nonzero()], 95)
        if vmin is None:
            vmin = M.min()
        
        if log:
            vmin = M[np.nonzero(M)].min()
            vmax = M.max()
            sc = h_ax.pcolormesh(x, y, np.flipud(M), cmap=cmap,
                        edgecolor='none', snap=True, linewidth=.001, norm=LogNorm(vmin, vmax), rasterized=True)
        else:
            sc = h_ax.pcolormesh(x, y, np.flipud(M), vmin=vmin, vmax=vmax, cmap=cmap,
                        edgecolor='none', snap=True, linewidth=.001, rasterized=True)
        
        h_ax.axis('off')
        self.heatmap_ax = h_ax
        self.hx = x
        self.hy = y
        
        # colorbar
        if not no_colorbar:
            c_ax = self.fig.add_axes([heatmap_pos[0]-0.02,
                                 (heatmap_pos[1]+0.9)/2,
                                 cbr_width,
                                 cbr_height])
            cbar = self.fig.colorbar(sc, cax=c_ax, ticks=[vmin, vmax], format='%.3g')
            cbar.outline.set_linewidth(0.3)
            c_ax.tick_params(labelsize=cbr_fontsize, length=0, pad=0.4)
            self.cbar_ax = c_ax

    def plot_coordinates(self, List, chrom_size=5, labelsize=5):

        xticks = []
        labels = []
        for p in List:
            if p > self.start and p < self.end:
                xticks.append((p - self.start) // self.res)
                labels.append('{0:,}'.format(p))
        
        ax = self.fig.add_subplot(self.grid[self.track_count])
        self.track_count += 1

        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, fontsize=labelsize)

        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        
        ax.tick_params(axis='both', bottom=False, top=True, left=False, right=False, direction='in', pad=-labelsize*1.8,
            labelbottom=False, labeltop=True, labelleft=False, labelright=False, width=0.5, length=2.5)

        ax.set_xlim(self.hx.min()-0.5, self.hx.max()+0.5)
        ax.set_ylim(0, 1)
        ax.text(-1, 0.8, self.chrom, fontsize=chrom_size, va='top', ha='center')
    
    def plot_signal(self, track_name, bw_fil, color='#666666', data_range_size=4,
        max_value='auto', min_value='auto', y_axis_offset=0.01, label_size=4, nBins=800):
        '''
        Choices for data_range_style: ['y-axis', 'text'].
        '''
        import pyBigWig

        db = pyBigWig.open(bw_fil)
        arr = np.array(db.stats(self.chrom, self.start, self.end, nBins=nBins)).astype(float)
        arr[np.isnan(arr)] = 0

        label_ax = self.fig.add_subplot(self.grid[self.track_count])
        self.track_count += 1
        ax = self.fig.add_subplot(self.grid[self.track_count])
        
        label_ax.text(0, 0.5, track_name, fontsize=label_size, ha='left', va='center')
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(0, 1)
        self.clear_frame(label_ax)

        ax.fill_between(np.arange(arr.size), arr, color=color, edgecolor='none')
        ax.set_xlim(0, arr.size-1)
        self.clear_frame(ax)
        ax_pos = ax.get_position().bounds
        y_ax = self.fig.add_axes([ax_pos[0]-y_axis_offset, ax_pos[1],
                                  y_axis_offset, ax_pos[3]])
        if min_value=='auto':
            min_value = 0
        if max_value=='auto':
            max_value = np.max(arr)

        plot_y_axis(y_ax, min_value, max_value, size=data_range_size)
        self.clear_frame(y_ax)

    
    def clear_frame(self, ax):

        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        
        ax.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
    def plot_SV(self, sv_fil, prob_cutoff=0.5, color='k', linestyle=':',
        marker='o', size=15, face_color='none', linewidths=0.5, alpha=1):

        Bool = np.zeros(self.matrix.shape, dtype=bool)
        SVs = load_sv_full(sv_fil)
        for c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2, ng in SVs:
            if (c1 == c2 == self.chrom) and (res2 <= self.res) and (max(prob1, prob2, prob3, prob4, prob5, prob6) > prob_cutoff):
                if self.start < p1 < p2 < self.end:
                    si = p1 // self.res - self.start // self.res
                    ei = p2 // self.res - self.start // self.res
                    Bool[si, ei] = 1

        lx = self.hx[:-1,:-1][np.flipud(Bool)]
        ly = self.hy[:-1,:-1][np.flipud(Bool)] + 1
        self.heatmap_ax.scatter(lx, ly, color=color, linestyle=linestyle, marker=marker, s=size, fc=face_color,
                            linewidths=linewidths, alpha=alpha)
        self.heatmap_ax.set_xlim(self.hx.min(), self.hx.max())
        self.heatmap_ax.set_ylim(self.hy.min(), self.hy.max())

    def outfig(self, outfile, dpi=200, bbox_inches='tight'):

        self.fig.savefig(outfile, dpi=dpi, bbox_inches=bbox_inches)
    
    def show(self):

        self.fig.show()


class interChrom(object):

    def __init__(self, uri, chrom_list, correct=False, figsize=(2.1, 1.4),
        n_rows=3, track_partition=[4,0.5,0.8], space=0.07):

        self.clr = cooler.Cooler(uri)
        self.res = self.clr.binsize
        
        # read chromosome sizes
        L = []
        coords = []
        for c in chrom_list:
            bin_starts = self.clr.bins().fetch(c)['start'].values
            L.append(bin_starts.size)
            for s in bin_starts:
                coords.append((c, s))
        self.coords_map = dict(zip(coords, range(len(coords))))

        cumsum = np.cumsum(L)
        cumsum = [0] + list(cumsum)

        # construct the genome-wide matrix
        big = np.zeros((cumsum[-1], cumsum[-1]))
        for i in range(len(chrom_list)):
            c1 = chrom_list[i]
            M1 = self.clr.matrix(balance=correct).fetch(c1)
            big[cumsum[i]:cumsum[i+1], cumsum[i]:cumsum[i+1]] = M1
            for j in range(len(chrom_list)):
                if i==j:
                    continue
                c2 = chrom_list[j]
                M3 = self.clr.matrix(balance=correct).fetch(c1, c2)
                M4 = self.clr.matrix(balance=correct).fetch(c2, c1)
                big[cumsum[i]:cumsum[i+1], cumsum[j]:cumsum[j+1]] = M3
                big[cumsum[j]:cumsum[j+1], cumsum[i]:cumsum[i+1]] = M4

        big[np.isnan(big)] = 0
        self.matrix = big

        # calculate chromosome bounds
        self.bounds = []
        for i in range(len(chrom_list)):
            self.bounds.append([cumsum[i], cumsum[i+1]])
        self.chrom_list = chrom_list

        # figure layout
        fig = plt.figure(figsize=figsize)
        self.fig = fig
        self.grid = GridSpec(n_rows, 1, figure=fig, left=0.1, right=0.9,
                    bottom=0.1, top=0.9, hspace=space, height_ratios=track_partition)
        self.track_count = 0

        # define my colormap (traditional w --> r)
        self.cmap = LinearSegmentedColormap.from_list('interaction',
                ['#FFFFFF','#FFDFDF','#FF7575','#FF2626','#F70000'])
    
    def matrix_plot(self, colormap='traditional', vmin=None, vmax=None, log=False,
        cbr_width=0.03, cbr_height=0.18, cbr_fontsize=4, no_colorbar=False):

        h_ax = self.fig.add_subplot(self.grid[self.track_count])
        self.track_count += 1

        heatmap_pos = h_ax.get_position().bounds

        M = self.matrix
        n = M.shape[0]

        # Create the rotation matrix
        t = np.array([[1,0.5], [-1,0.5]])
        A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)

        if colormap=='traditional':
            cmap = self.cmap
        else:
            cmap = colormap
        
        # Plot the Heatmap ...
        x = A[:,1].reshape(n+1, n+1)
        y = A[:,0].reshape(n+1, n+1)
        y[y<0] = -y[y<0]

        if vmax is None:
            vmax = np.percentile(M[M.nonzero()], 95)
        if vmin is None:
            vmin = M.min()
        
        if log:
            vmin = M[np.nonzero(M)].min()
            vmax = M.max()
            sc = h_ax.pcolormesh(x, y, np.flipud(M), cmap=cmap,
                        edgecolor='none', snap=True, linewidth=.001, norm=LogNorm(vmin, vmax), rasterized=True)
        else:
            sc = h_ax.pcolormesh(x, y, np.flipud(M), vmin=vmin, vmax=vmax, cmap=cmap,
                        edgecolor='none', snap=True, linewidth=.001, rasterized=True)
        
        h_ax.axis('off')
        self.heatmap_ax = h_ax
        self.hx = x
        self.hy = y
        
        # colorbar
        if not no_colorbar:
            c_ax = self.fig.add_axes([heatmap_pos[0]-0.02,
                                 (heatmap_pos[1]+0.9)/2,
                                 cbr_width,
                                 cbr_height])
            cbar = self.fig.colorbar(sc, cax=c_ax, ticks=[vmin, vmax], format='%.3g')
            cbar.outline.set_linewidth(0.3)
            c_ax.tick_params(labelsize=cbr_fontsize, length=0, pad=0.4)
            self.cbar_ax = c_ax
    
    def plot_SV(self, sv_fil, prob_cutoff=0.5, color='k', linestyle=':',
        marker='o', size=15, face_color='none', linewidths=0.5, alpha=1):

        Bool = np.zeros(self.matrix.shape, dtype=bool)
        SVs = load_sv_full(sv_fil)
        for c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2, ng in SVs:
            if (c1 != c2) and (res2 <= self.res) and (max(prob1, prob2, prob3, prob4, prob5, prob6) > prob_cutoff):
                    loci1 = (c1, p1//self.res*self.res)
                    loci2 = (c2, p2//self.res*self.res)
                    if (loci1 in self.coords_map) and (loci2 in self.coords_map):
                        si = self.coords_map[loci1]
                        ei = self.coords_map[loci2]
                        if si > ei:
                            si, ei = ei, si
                        Bool[si, ei] = 1

        lx = self.hx[:-1,:-1][np.flipud(Bool)]
        ly = self.hy[:-1,:-1][np.flipud(Bool)] + 1
        self.heatmap_ax.scatter(lx, ly, color=color, linestyle=linestyle, marker=marker, s=size, fc=face_color,
                            linewidths=linewidths, alpha=alpha)
        self.heatmap_ax.set_xlim(self.hx.min(), self.hx.max())
        self.heatmap_ax.set_ylim(self.hy.min(), self.hy.max())
    
    def plot_chromosome_bounds(self, line_color='k', linewidth=0.4, linestype='-'):

        n = self.matrix.shape[0]

        for si, ei in self.bounds:
            if ei > n - 1:
                ei = n - 1
            
            x = [self.hx[:-1, :-1][n-1-si, si],
                 self.hx[:-1, :-1][n-1-si, ei],
                 self.hx[:-1, :-1][n-1-ei, ei],
                 self.hx[:-1, :-1][n-1-si, si]]
            y = [self.hy[:-1, :-1][n-1-si, si] - 1,
                 self.hy[:-1, :-1][n-1-si, ei] + 1,
                 self.hy[:-1, :-1][n-1-ei, ei] - 1,
                 self.hy[:-1, :-1][n-1-si, si] - 1]
            self.heatmap_ax.plot(x, y, color=line_color, linestyle=linestype,
                linewidth=linewidth)
        
        self.heatmap_ax.set_xlim(self.hx.min(), self.hx.max())
        self.heatmap_ax.set_ylim(self.hy.min()-1, self.hy.max())
    
    def add_chrom_labels(self, labelsize=5):

        ax = self.fig.add_subplot(self.grid[self.track_count])
        self.track_count += 1

        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        
        ax.axis('off')

        ax.set_xlim(self.hx.min()-0.5, self.hx.max()+0.5)
        ax.set_ylim(0, 1)
        for i in range(len(self.chrom_list)):
            si, ei = self.bounds[i]
            ax.text((si+ei)/2, 1, self.chrom_list[i], ha='center', va='top', fontsize=labelsize)
    
    def outfig(self, outfile, dpi=200, bbox_inches='tight'):

        self.fig.savefig(outfile, dpi=dpi, bbox_inches=bbox_inches)
    
    def show(self):

        self.fig.show()


def plot_y_axis(y_ax, ymin, ymax, size):
    
    def value_to_str(value):
        if value % 1 == 0:
            str_value = str(int(value))
        else:
            if value < 0.01:
                str_value = "{:.4f}".format(value)
            else:
                str_value = "{:.2f}".format(value)
        return str_value

    ymax_str = value_to_str(ymax)
    ymin_str = value_to_str(ymin)
    x_pos = [0, 0.5, 0.5, 0]
    y_pos = [0.01, 0.01, 0.99, 0.99]
    y_ax.plot(x_pos, y_pos, color='black', linewidth=1, transform=y_ax.transAxes)
    y_ax.text(-0.2, -0.01, ymin_str, verticalalignment='bottom', horizontalalignment='right',
                transform=y_ax.transAxes, fontsize=size)
    y_ax.text(-0.2, 1, ymax_str, verticalalignment='top', horizontalalignment='right',
                transform=y_ax.transAxes, fontsize=size)
    y_ax.patch.set_visible(False)