import logging, hdbscan
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from joblib import Parallel, delayed
from collections import defaultdict
from eaglec.utilities import local_background

log = logging.getLogger(__name__)

def apply_buff(candi):

    expanded = set()
    for xi, yi, buf in candi:
        for i in range(-buf, buf+1):
            for j in range(-buf, buf+1):
                expanded.add((xi+i, yi+j))

    return expanded

def iterative_clustering(coords, values, min_cluster_size=4, min_samples=4,
                         max_cluster_size=100, cut_ratio=0.8, round=3):
    
    for _ in range(round):
        x, y = coords.T
        nx = []
        ny = []
        nv = []
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples).fit(coords)
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_, v_ = x[mask], y[mask], values[mask]
            if (ci == -1) or (x_.size < max_cluster_size):
                nx.extend(list(x_))
                ny.extend(list(y_))
                nv.extend(list(v_))
            else:
                n = int(np.ceil(x_.size * cut_ratio))
                sort_table = []
                for xi, yi, vi in zip(x_, y_, v_):
                    sort_table.append((vi, xi, yi))
                
                sort_table.sort(reverse=True)
                for vi, xi, yi in sort_table[:n]:
                    nx.append(xi)
                    ny.append(yi)
                    nv.append(vi)
        
        coords = np.r_[list(zip(nx, ny))]
        values = np.r_[nv]
    
    return x, y, clusterer


def filter_intra_singletons(M, nM, weights, exp, x, y, thre=0.02, ww=3, pw=1):

    mask = (x - ww >= 0) & (x + ww + 1 <= M.shape[0]) & \
           (y - ww >= 0) & (y + ww + 1 <= M.shape[1])
    x, y = x[mask], y[mask]
    x_filtered = []
    y_filtered = []
    if x.size > 0:
        seed = np.arange(-ww, ww+1)
        delta = np.tile(seed, (seed.size, 1))
        xxx = x.reshape((x.size, 1, 1)) + delta.T
        yyy = y.reshape((y.size, 1, 1)) + delta
        v = np.array(nM[xxx.ravel(), yyy.ravel()]).ravel()
        vvv = v.reshape((x.size, seed.size, seed.size))
        for i in range(x.size):
            xi = x[i]
            yi = y[i]
            window = vvv[i].astype(exp.dtype)
            window[np.isnan(window)] = 0
            window[(ww-pw):(ww+pw+1), (ww-pw):(ww+pw+1)] = 0

            nonzero = window.nonzero()[0]
            if nonzero.size / window.size < 0.08:
                continue

            E_ = local_background(window, exp, xi, yi, ww)
            if not weights is None:
                evalue = E_ / (weights[xi] * weights[yi])
            else:
                evalue = E_
            
            Poiss = stats.poisson(evalue)
            pvalue = Poiss.sf(M[xi, yi])
            if pvalue < thre:
                x_filtered.append(xi)
                y_filtered.append(yi)
    
    x_filtered = np.r_[x_filtered]
    y_filtered = np.r_[y_filtered]
    
    return x_filtered, y_filtered

def filter_intra_cluster_points(M, nM, weights, exp, x, y, pw=1, min_points=10):
    
    dis = []
    expanded_coords = set()
    for xi, yi in zip(x, y):
        for i in range(-pw, pw+1):
            for j in range(-pw, pw+1):
                if (xi+i > 0) and (xi+i < M.shape[0]) and (yi+j > 0) and (yi+j < M.shape[1]):
                    expanded_coords.add((xi+i, yi+j))
                    d = (yi+j) - (xi+i)
                    dis.append(d)

    coords = set()
    for xi, yi in expanded_coords:
        v = nM[xi, yi]
        if np.isnan(v):
            continue
        if v == 0:
            continue
        coords.add((xi, yi))
    
    # correct the peak width for narrow clusters
    if (x.max() - x.min() < (2*pw+1)) or (y.max() - y.min() < (2*pw+1)):
        pw = 0

    # calculate the p-values
    maxdis = max(dis)
    mindis = min(dis)
    singletons = set()
    filtered_table = []
    for xi, yi in zip(x, y):
        peaks = set()
        for i in range(-pw, pw+1):
            for j in range(-pw, pw+1):
                peaks.add((xi+i, yi+j))
        
        bg_coords = coords - peaks
        if len(bg_coords) >= min_points:
            x, y = np.r_[list(bg_coords)].T
            if maxdis >= exp.size:
                E_ = nM[x, y].mean()
            else:
                Ed = exp[y-x]
                E_ = nM[x, y].sum() / Ed.sum() * exp[yi-xi]

            if not weights is None:
                evalue = E_ / (weights[xi] * weights[yi])
            else:
                evalue = E_
    
            Poiss = stats.poisson(evalue)
            pvalue = Poiss.sf(M[xi, yi])
            filtered_table.append((pvalue, xi, yi))
        else:
            singletons.add((xi, yi))
    
    short_range = True
    if mindis > exp.size:
        short_range = False

    return filtered_table, singletons, short_range

def select_intra_core(clr, c, balance, Ed, k=100, q_thre=0.01, minv=1, min_cluster_size=4,
                      min_samples=4, max_cluster_size=250, decay_rate=0.8, niter=3, buff=2,
                      highres=False):

    M = clr.matrix(balance=False, sparse=True).fetch(c).tocsr()
    x, y = M.nonzero()
    v = M.data
    # check for long-range contacts
    dis = y - x
    if highres:
        mask = (dis >= k) & (dis <= Ed.size//4) & (v > minv)
    else:
        mask = (dis >= k) & (v > minv)
        
    x, y, v = x[mask], y[mask], v[mask]
    evalue = Ed[k]
    if balance:
        nM = clr.matrix(balance=balance, sparse=True).fetch(c).tocsr()
        weights = clr.bins().fetch(c)[balance].values
        b1 = weights[x]
        b2 = weights[y]
        evalue = evalue / (b1 * b2)
        mask = np.isfinite(evalue)
        evalue = evalue[mask]
        x = x[mask]
        y = y[mask]
        v = v[mask]
        if evalue.size > 0:
            Poiss = stats.poisson(evalue)
            pvalues = Poiss.sf(v)
        else:
            pvalues = np.array([], dtype=float)
    else:
        nM = M
        weights = None
        Poiss = stats.poisson(evalue)
        pvalues = Poiss.sf(v)

    # check for short-range contacts
    idx = np.arange(M.shape[0])
    for i in range(10, k):
        diag = M.diagonal(i)
        xi = idx[:-i][diag>minv]
        yi = idx[i:][diag>minv]
        diag = diag[diag>minv]
        if diag.size > 0:
            if not balance:
                x = np.r_[x, xi]
                y = np.r_[y, yi]
                Poiss = stats.poisson(Ed[i])
                pvalues = np.r_[pvalues, Poiss.sf(diag)]
            else:
                b1 = weights[xi]
                b2 = weights[yi]
                evalue = Ed[i] / (b1 * b2)
                mask = np.isfinite(evalue)
                evalue = evalue[mask]
                xi = xi[mask]
                yi = yi[mask]
                diag = diag[mask]
                if diag.size > 0:
                    x = np.r_[x, xi]
                    y = np.r_[y, yi]
                    Poiss = stats.poisson(evalue)
                    pvalues = np.r_[pvalues, Poiss.sf(diag)]

    qvalues = multipletests(pvalues.ravel(), method='fdr_bh')[1]
    mask = qvalues < q_thre
    x, y = x[mask], y[mask]

    candi = set()
    cutoff = 0.05 # hard-coded param
    buf = buff - 1
    if (min_cluster_size > 0) and (min_samples > 0) and (x.size > min_samples):
        singletons = set()
        # iterative clustering
        coords = np.r_['1,2,0', x, y]
        values = np.array(nM[x, y]).ravel()
        exp = np.zeros(values.size)
        dis = y - x
        exp[dis>=Ed.size] = Ed[-1]
        exp[dis<Ed.size] = Ed[dis[dis<Ed.size]]
        values = values / exp
        x, y, clusterer = iterative_clustering(coords, values, min_cluster_size=min_cluster_size,
                                               min_samples=min_samples, max_cluster_size=max_cluster_size,
                                               cut_ratio=decay_rate, round=niter)
        coords = np.r_[list(zip(x, y))]
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            if ci >= 0:
                filtered_table, tmp, short_range = filter_intra_cluster_points(M, nM, weights, Ed, x_, y_)
                singletons.update(tmp)
                for pv, xi, yi in filtered_table:
                    if pv < cutoff:
                        if short_range:
                            candi.add((xi, yi, buff))
                        else:
                            candi.add((xi, yi, buf))
            else:
                tmp = set(zip(x_, y_))
                singletons.update(tmp)

        if len(singletons) > 0:
            x_, y_ = np.r_[list(singletons)].T
            x_f, y_f = filter_intra_singletons(M, nM, weights, Ed, x_, y_, thre=cutoff)
            for xi, yi in zip(x_f, y_f):
                if yi - xi <= Ed.size:
                    candi.add((xi, yi, buff))
                else:
                    candi.add((xi, yi, buf))
    else:
        for xi, yi in zip(x, y):
            candi.add((xi, yi, buf))
    
    candi = apply_buff(candi)

    return c, candi

def select_intra_candidate(clr, chroms, balance, Ed, k=100, q_thre=0.01, minv=1,
                           min_cluster_size=3, min_samples=3, max_cluster_size=250,
                           decay_rate=0.8, niter=3, buff=2, nproc=4, highres=False):

    queue = []
    for c in chroms:
        queue.append((clr, c, balance, Ed[c], k, q_thre, minv, min_cluster_size,
                      min_samples, max_cluster_size, decay_rate, niter, buff,
                      highres))
    
    results = Parallel(n_jobs=nproc)(delayed(select_intra_core)(*i) for i in queue)
    bychrom = {}
    for c, candi in results:
        if len(candi):
            bychrom[(c, c)] = candi
    
    return bychrom

def generage_bin_edges(axis_size, w):

    ws = 2 * w + 1
    bin_edge = [0]
    for i in range(axis_size//ws+1):
        bin_start = bin_edge[-1]
        bin_end = min(bin_start+ws, axis_size)
        if bin_start < axis_size:
            bin_edge.append(bin_end)
        if bin_end >= axis_size:
            break

    return bin_edge

def filter_candidates(candidate_pool):

    candidates = candidate_pool[0]
    for tmp in candidate_pool[1:]:
        candidates = candidates & tmp
    
    return candidates

def add_neighbor_candidates(candidates, M, buff=1):

    new_candidates = candidates.copy()
    for xi, yi in candidates:
        ref = M[xi, yi]
        for i in range(-buff, buff+1):
            for j in range(-buff, buff+1):
                x = xi + i
                y = yi + j
                if (0 < x < M.shape[0]) and (0 < y < M.shape[1]):
                    if M[x, y] > ref:
                        new_candidates.add((x, y))
    
    return new_candidates

def remove_real_outliers(x, y, buff=1):

    pool = set(zip(x, y))
    filtered = set()
    for xi, yi in pool:
        for i in range(-buff, buff+1):
            for j in range(-buff, buff+1):
                if (i == 0) and (j == 0):
                    continue
                if (xi+i, yi+j) in pool:
                    filtered.add((xi, yi))
    
    return filtered

def filter_inter_singletons(M, nM, weights_1, weights_2, x, y, thre=0.02, ww=3, pw=1):

    mask = (x - ww >= 0) & (x + ww + 1 <= M.shape[0]) & \
           (y - ww >= 0) & (y + ww + 1 <= M.shape[1])
    x, y = x[mask], y[mask]
    x_filtered = []
    y_filtered = []
    if x.size > 0:
        seed = np.arange(-ww, ww+1)
        delta = np.tile(seed, (seed.size, 1))
        xxx = x.reshape((x.size, 1, 1)) + delta.T
        yyy = y.reshape((y.size, 1, 1)) + delta
        v = np.array(nM[xxx.ravel(), yyy.ravel()]).ravel()
        vvv = v.reshape((x.size, seed.size, seed.size))
        for i in range(x.size):
            xi = x[i]
            yi = y[i]
            window = vvv[i]
            window[np.isnan(window)] = 0
            window[(ww-pw):(ww+pw+1), (ww-pw):(ww+pw+1)] = 0

            nonzero = window.nonzero()
            if nonzero[0].size / window.size < 0.08:
                continue

            E_ = window[nonzero].mean()
            if not weights_1 is None:
                evalue = E_ / (weights_1[xi] * weights_2[yi])
            else:
                evalue = E_
            
            Poiss = stats.poisson(evalue)
            pvalue = Poiss.sf(M[xi, yi])
            if pvalue < thre:
                x_filtered.append(xi)
                y_filtered.append(yi)
    
    x_filtered = np.r_[x_filtered]
    y_filtered = np.r_[y_filtered]
    
    return x_filtered, y_filtered

def filter_inter_cluster_points(M, nM, weights_1, weights_2, x, y, pw=1, min_points=10):

    coords = set()
    for xi, yi in zip(x, y):
        for i in range(-pw, pw+1):
            for j in range(-pw, pw+1):
                if (xi+i > 0) and (xi+i < M.shape[0]) and (yi+j > 0) and (yi+j < M.shape[1]):
                    v = nM[xi+i, yi+j]
                    if np.isnan(v):
                        continue
                    if v == 0:
                        continue
                    coords.add((xi+i, yi+j))
    
    # correct the peak width for narrow clusters
    if (x.max() - x.min() < (2*pw+1)) or (y.max() - y.min() < (2*pw+1)):
        pw = 0

    # calculate the p-values
    singletons = set()
    results = []
    for xi, yi in zip(x, y):
        peaks = set()
        for i in range(-pw, pw+1):
            for j in range(-pw, pw+1):
                peaks.add((xi+i, yi+j))
        
        bg_coords = coords - peaks
        if len(bg_coords) >= min_points:
            x, y = np.r_[list(bg_coords)].T
            E_ = nM[x, y].mean()

            if not weights_1 is None:
                evalue = E_ / (weights_1[xi] * weights_2[yi])
            else:
                evalue = E_
    
            Poiss = stats.poisson(evalue)
            pvalue = Poiss.sf(M[xi, yi])
            results.append((pvalue, xi, yi))
        else:
            singletons.add((xi, yi))
    
    return results, singletons

def filter_inter_total(M, nM, weights_1, weights_2, x, y):

    results = []
    # avoid estimating the expected value using too many points
    if not weights_1 is None:
        narr = np.array(nM[x, y]).ravel()
        mask = np.isfinite(narr)
        if mask.sum() == 0:
            return results
        narr, x, y = narr[mask], x[mask], y[mask]
        varr = np.array(M[x, y]).ravel()
    else:
        narr = np.array(nM[x, y]).ravel()
        varr = narr
    
    avg = narr.mean()
    
    # calculate the p-values
    if not weights_1 is None:
        evalues = avg / (weights_1[x] * weights_2[y])
    else:
        evalues = avg
    
    Poiss = stats.poisson(evalues)
    pvalues = Poiss.sf(varr)
    results = list(zip(pvalues, x, y))
    
    return results
    
def select_inter_core(clr, c1, c2, balance, windows, min_per, q_thre=0.01,
                      min_cluster_size=4, min_samples=4, max_cluster_size=100,
                      decay_rate=0.6, niter=3, buff=1):

    M = clr.matrix(balance=False, sparse=True).fetch(c1, c2).tocsr()
    x, y = M.nonzero()
    v = M.data
    
    candidates_pool = []
    # all contacts are treated as a single background
    for w in windows:
        x_edge = generage_bin_edges(M.shape[0], w)
        y_edge = generage_bin_edges(M.shape[1], w)
        sum_v, x_edge, y_edge, bin_indices = stats.binned_statistic_2d(
            x, y, v, statistic='sum', bins=(x_edge, y_edge), expand_binnumbers=True
        )

        l_v = np.percentile(sum_v[sum_v>0], min_per)
        mask = sum_v > l_v
        if mask.sum() > 100:
            evalue = sum_v[mask].mean()
        else:
            evalue = sum_v[sum_v>0].mean()

        Poiss = stats.poisson(evalue)
        pvalues = Poiss.sf(sum_v)
        qvalues = multipletests(pvalues.ravel(), method='fdr_bh')[1]
        qvalues = qvalues.reshape(pvalues.shape)
        boolM = qvalues < q_thre

        bin_indices = bin_indices - 1
        bool_arr = boolM[bin_indices[0], bin_indices[1]]
        xs, ys = x[bool_arr], y[bool_arr]
        coords = set(zip(xs, ys))

        candidates_pool.append(coords)
    
    candidates_pool = filter_candidates(candidates_pool)
    candidates_pool = add_neighbor_candidates(candidates_pool, M)

    if balance:
        nM = clr.matrix(balance=balance, sparse=True).fetch(c1, c2).tocsr()
        weights_1 = clr.bins().fetch(c1)[balance].values
        weights_2 = clr.bins().fetch(c2)[balance].values
    else:
        nM = M
        weights_1 = None
        weights_2 = None

    candi = set()
    cutoff = 0.05
    if (min_cluster_size > 0) and (min_samples > 0) and (len(candidates_pool) > min_samples):
        coords = np.r_[list(candidates_pool)]
        x, y = coords.T
        table = filter_inter_total(M, nM, weights_1, weights_2, x, y)
        x = []
        y = []
        for pv, xi, yi in table:
            if pv < q_thre:
                x.append(xi)
                y.append(yi)
        
        if len(x) > min_samples:
            singletons = set()
            x = np.r_[x]
            y = np.r_[y]
            coords = np.r_[list(zip(x, y))]
            values = np.array(nM[x, y]).ravel()
            # first round of clustering
            x, y, clusterer = iterative_clustering(coords, values, min_cluster_size=min_cluster_size,
                                                min_samples=min_samples, max_cluster_size=max_cluster_size,
                                                cut_ratio=decay_rate, round=niter)
            coords = np.r_[list(zip(x, y))]
            for ci in set(clusterer.labels_):
                mask = clusterer.labels_ == ci
                x_, y_ = x[mask], y[mask]
                if ci >= 0:
                    table, tmp= filter_inter_cluster_points(M, nM, weights_1, weights_2, x_, y_)
                    singletons.update(tmp)
                    for pv, xi, yi in table:
                        if pv < cutoff:
                            candi.add((xi, yi, buff))
                else:
                    tmp = set(zip(x_, y_))
                    singletons.update(tmp)

            if len(singletons) > 0:
                x_, y_ = np.r_[list(singletons)].T
                x_f, y_f = filter_inter_singletons(M, nM, weights_1, weights_2, x_, y_,
                                                   thre=cutoff)
                for xi, yi in zip(x_f, y_f):
                    candi.add((xi, yi, buff))
        else:
            for xi, yi in candidates_pool:
                candi.add((xi, yi, buff))
    else:
        for xi, yi in candidates_pool:
            candi.add((xi, yi, buff))
    
    candi = apply_buff(candi)
    
    return c1, c2, candi

def select_inter_candidate(clr, chroms, balance, windows=[3,5], min_per=50,
                           q_thre=0.01, min_cluster_size=4, min_samples=4,
                           max_cluster_size=100, decay_rate=0.6, niter=3, buff=1,
                           nproc=4):
    
    queue = []
    for i in range(len(chroms)-1):
        for j in range(i+1, len(chroms)):
            queue.append((clr, chroms[i], chroms[j], balance, windows, min_per,
                          q_thre, min_cluster_size, min_samples, max_cluster_size,
                          decay_rate, niter, buff))
    
    results = Parallel(n_jobs=nproc)(delayed(select_inter_core)(*i) for i in queue)
    bychrom = {}
    for c1, c2, candi in results:
        if len(candi):
            bychrom[(c1, c2)] = candi
    
    return bychrom

def cross_resolution_filter(byres, byres_bad, min_dis=50):

    new = {}
    for tr in byres:
        new[tr] = defaultdict(set)
        for c in byres[tr]:
            for tx, ty in byres[tr][c]:
                if (c[0] == c[1]) and (ty - tx < min_dis):
                    new[tr][c].add((tx, ty))
                else:
                    valid = True
                    for qr in byres_bad:
                        if (qr > tr) and (c in byres_bad[qr]):
                            s_l = range(tx*tr//qr, int(np.ceil((tx+1)*tr/qr)))
                            e_l = range(ty*tr//qr, int(np.ceil((ty+1)*tr/qr)))
                            for qx in s_l:
                                for qy in e_l:
                                    if (c[0] == c[1]) and (qy - qx < min_dis):
                                        continue
                                    if (qx, qy) in byres_bad[qr][c]:
                                        valid = False
                                        break
                    
                    if valid:
                        new[tr][c].add((tx, ty))
    
    return new