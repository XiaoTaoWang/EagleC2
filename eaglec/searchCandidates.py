import logging, hdbscan
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from joblib import Parallel, delayed
from collections import defaultdict

log = logging.getLogger(__name__)

def _remove_real_outliers(x, y, buff=1):

    pool = set(zip(x, y))
    filtered = set()
    for xi, yi in pool:
        for i in range(-buff, buff+1):
            for j in range(-buff, buff+1):
                if (xi+i, yi+j) in pool:
                    filtered.add((xi, yi))
    
    return filtered

def check_neighbor_signals(M, x, y):

    signals = []
    shifts = [(1, 0), (0, 1), (-1, 0), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i, j in shifts:
        xi = x + i
        yi = y + j
        if (xi >= 0) and (xi < M.shape[0]) and (yi >= 0) and (yi < M.shape[1]):
            v = M[xi, yi]
            if np.isnan(v):
                continue
            signals.append(v)
    
    return signals

def select_intra_core(clr, c, balance, Ed, k=100, q_thre=0.01, minv=1, min_cluster_size=3,
                      min_samples=3, shrink_per=30, top_per=10, top_n=10, buff=2,
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

    if balance:
        M = clr.matrix(balance=balance, sparse=True).fetch(c).tocsr()

    candi = set()
    bad_pixels = set()
    top_per = top_per / 100
    max_cluster_size = 100 # hard-coded param
    filter_min_width = 5 # hard-coded param
    filter_top_per = 0.15 # hard-coded param
    filter_top_n = 15 # hard-coded param
    filter_min_cluster_size = 20 # hard-coded param
    cut_ratio = shrink_per / 100
    buf = buff - 1
    if (min_cluster_size > 0) and (min_samples > 0) and (x.size > min_samples):
        # first round of clustering
        coords = np.r_['1,2,0', x, y]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples).fit(coords)
        nx = []
        ny = []
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            if (ci == -1) or (x_.size < max_cluster_size):
                nx.extend(list(x_))
                ny.extend(list(y_))
            else:
                n = int(np.ceil(x_.size * cut_ratio))
                sort_table = []
                for xi, yi in zip(x_, y_):
                    v = M[xi, yi]
                    if np.isnan(v):
                        continue
                    d = yi - xi
                    if d + 1 > Ed.size:
                        v = v / Ed[-1]
                    else:
                        v = v / Ed[d]
                    sort_table.append((v, xi, yi))
                
                sort_table.sort(reverse=True)
                for _, xi, yi in sort_table[:n]:
                    nx.append(xi)
                    ny.append(yi)
        
        # second round of clustering
        x = np.r_[nx]
        y = np.r_[ny]
        coords = np.r_['1,2,0', x, y]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples).fit(coords)
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            if ci == -1:
                tmp = _remove_real_outliers(x_, y_)
                for xi, yi in tmp:
                    for i in range(-buf, buf+1):
                        for j in range(-buf, buf+1):
                            candi.add((xi+i, yi+j))
                continue
            else:
                if x_.size < top_n:
                    for xi, yi in zip(x_, y_):
                        for i in range(-buff, buff+1):
                            for j in range(-buff, buff+1):
                                candi.add((xi+i, yi+j))
                else:
                    n = min(top_n, int(np.ceil(x_.size * top_per)))
                    sort_table = []
                    for xi, yi in zip(x_, y_):
                        v = M[xi, yi]
                        if np.isnan(v):
                            continue
                        d = yi - xi
                        if d + 1 > Ed.size:
                            v = v / Ed[-1]
                        else:
                            v = v / Ed[d]
                        sort_table.append((v, xi, yi))

                    sort_table.sort(reverse=True)
                    for _, xi, yi in sort_table[:n]:
                        for i in range(-buff, buff+1):
                            for j in range(-buff, buff+1):
                                candi.add((xi+i, yi+j))
                    
                    if (x_.max() - x_.min() > filter_min_width) and (y_.max() - y_.min() > filter_min_width) and \
                       (x_.size > filter_min_cluster_size):
                        fn = min(filter_top_n, int(np.ceil(x_.size * filter_top_per)))
                        for _, xi, yi in sort_table[fn:]:
                            for i in range(-buff, buff+1):
                                for j in range(-buff, buff+1):
                                    bad_pixels.add((xi+i, yi+j))
    else:
        for xi, yi in zip(x, y):
            for i in range(-buf, buf+1):
                for j in range(-buf, buf+1):
                    candi.add((xi+i, yi+j))
    
    bad_pixels = bad_pixels - candi

    return c, candi, coords, clusterer

def select_intra_candidate(clr, chroms, balance, Ed, k=100, q_thre=0.01, minv=1,
                           min_cluster_size=3, min_samples=3, shrink_per=15,
                           top_per=10, top_n=10, buff=2, nproc=4, highres=False):

    queue = []
    for c in chroms:
        queue.append((clr, c, balance, Ed[c], k, q_thre, minv, min_cluster_size,
                      min_samples, shrink_per, top_per, top_n, buff,
                      highres))
    
    results = Parallel(n_jobs=nproc)(delayed(select_intra_core)(*i) for i in queue)
    bychrom = {}
    bychrom_bad = {}
    for c, candi, bad_pixels in results:
        if len(candi):
            bychrom[(c, c)] = candi
            bychrom_bad[(c, c)] = bad_pixels
    
    return bychrom, bychrom_bad

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

def select_inter_core(clr, c1, c2, balance, windows, min_per, q_thre=0.01,
                      min_cluster_size=3, min_samples=3, shrink_per=15,
                      top_per=10, top_n=10, buff=2):

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

    if balance:
        M = clr.matrix(balance=balance, sparse=True).fetch(c1, c2).tocsr()

    candi = set()
    bad_pixels = set()
    top_per = top_per / 100
    max_cluster_size = 100 # hard-coded param
    filter_min_width = 5 # hard-coded param
    filter_top_per = 0.15 # hard-coded param
    filter_top_n = 15 # hard-coded param
    filter_min_cluster_size = 20 # hard-coded param
    cut_ratio = shrink_per / 100
    buf = buff - 1
    if (min_cluster_size > 0) and (min_samples > 0) and (len(candidates_pool) > min_samples):
        # first round of clustering
        coords = np.r_[list(candidates_pool)]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples).fit(coords)
        nx = []
        ny = []
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_ = coords[mask].T
            if (ci == -1) or (x_.size < max_cluster_size):
                nx.extend(list(x_))
                ny.extend(list(y_))
            else:
                n = int(np.ceil(x_.size * cut_ratio))
                sort_table = []
                for xi, yi in zip(x_, y_):
                    v = M[xi, yi]
                    if np.isnan(v):
                        continue
                    sort_table.append((v, xi, yi))
                
                sort_table.sort(reverse=True)
                for _, xi, yi in sort_table[:n]:
                    nx.append(xi)
                    ny.append(yi)
        
        # second round of clustering
        x = np.r_[nx]
        y = np.r_[ny]
        coords = np.r_['1,2,0', x, y]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples).fit(coords)
        for ci in set(clusterer.labels_):
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            if ci == -1:
                tmp = _remove_real_outliers(x_, y_)
                for xi, yi in tmp:
                    for i in range(-buf, buf+1):
                        for j in range(-buf, buf+1):
                            candi.add((xi+i, yi+j))
                continue
            else:
                if x_.size < top_n:
                    for xi, yi in zip(x_, y_):
                        for i in range(-buff, buff+1):
                            for j in range(-buff, buff+1):
                                candi.add((xi+i, yi+j))
                else:
                    n = min(top_n, int(np.ceil(x_.size * top_per)))
                    sort_table = []
                    for xi, yi in zip(x_, y_):
                        v = M[xi, yi]
                        if np.isnan(v):
                            continue
                        sort_table.append((v, xi, yi))
                    
                    sort_table.sort(reverse=True)
                    for _, xi, yi in sort_table[:n]:
                        for i in range(-buff, buff+1):
                            for j in range(-buff, buff+1):
                                candi.add((xi+i, yi+j))
                    
                    if (x_.max() - x_.min() > filter_min_width) and (y_.max() - y_.min() > filter_min_width) and \
                       (x_.size > filter_min_cluster_size):
                        fn = min(filter_top_n, int(np.ceil(x_.size * filter_top_per)))
                        for _, xi, yi in sort_table[fn:]:
                            for i in range(-buff, buff+1):
                                for j in range(-buff, buff+1):
                                    bad_pixels.add((xi+i, yi+j))
    else:
        for xi, yi in candidates_pool:
            for i in range(-buf, buf+1):
                for j in range(-buf, buf+1):
                    candi.add((xi+i, yi+j))
    
    bad_pixels = bad_pixels - candi
    
    return c1, c2, candi, bad_pixels

def select_inter_candidate(clr, chroms, balance, windows=[3,4,5], min_per=50,
                           q_thre=0.01, min_cluster_size=3, min_samples=3,
                           shrink_per=15, top_per=10, top_n=10, buff=2, nproc=4):

    queue = []
    for i in range(len(chroms)-1):
        for j in range(i+1, len(chroms)):
            queue.append((clr, chroms[i], chroms[j], balance, windows, min_per,
                          q_thre, min_cluster_size, min_samples, shrink_per,
                          top_per, top_n, buff))
    
    results = Parallel(n_jobs=nproc)(delayed(select_inter_core)(*i) for i in queue)
    bychrom = {}
    bychrom_bad = {}
    for c1, c2, candi, bad_pixels in results:
        if len(candi):
            bychrom[(c1, c2)] = candi
            bychrom_bad[(c1, c2)] = bad_pixels
    
    return bychrom, bychrom_bad

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
                                    if (qx, qy) in byres_bad[qr][c]:
                                        valid = False
                                        break
                    
                    if valid:
                        new[tr][c].add((tx, ty))
    
    return new