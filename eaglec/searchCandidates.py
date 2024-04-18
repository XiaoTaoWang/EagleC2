import logging, hdbscan
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import defaultdict, Counter
from joblib import Parallel, delayed

log = logging.getLogger(__name__)

def concensus_clustering(coords, min_cluster_sizes=[3,4,5]):

    label_pool = []
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                    min_samples=mcs).fit(coords)
        label_pool.append(clusterer.labels_)
    
    for i in range(1, len(label_pool)):
        si = label_pool[i-1].max() + 1
        tmp_labels = label_pool[i]
        tmp_labels[tmp_labels>=0] += si
        label_pool[i] = tmp_labels
    
    label_pool = np.r_[label_pool]
    label_count = Counter(label_pool.ravel())
    labels_ = []
    for i in range(label_pool.shape[1]):
        table = []
        tmp = label_pool[:,i]
        for t in tmp:
            if t == -1:
                table.append((1, t))
            else:
                table.append((label_count[t], t))
        table.sort(reverse=True)
        labels_.append(table[0][1])
    
    labels_ = np.r_[labels_]
    
    return labels_


def select_intra_core(clr, c, Ed, k=100, q_thre=0.01, minv=1, min_cluster_size=3,
                      min_samples=3, shrink_per=15, top_per=10, top_n=10, buff=2,
                      highres=False):

    M = clr.matrix(balance=False, sparse=True).fetch(c).tocsr()
    x, y = M.nonzero()
    v = M.data
    # check for long-range contacts
    evalue = Ed[k]
    dis = y - x
    if highres:
        mask = (dis >= k) & (dis <= Ed.size//4) & (v > minv)
    else:
        mask = (dis >= k) & (v > minv)
        
    x, y, v = x[mask], y[mask], v[mask]
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
            x = np.r_[x, xi]
            y = np.r_[y, yi]
            Poiss = stats.poisson(Ed[i])
            pvalues = np.r_[pvalues, Poiss.sf(diag)]

    qvalues = multipletests(pvalues.ravel(), method='fdr_bh')[1]
    mask = qvalues < q_thre
    x, y = x[mask], y[mask]

    candi = set()
    top_per = top_per / 100
    max_cluster_size = 100 # hard-coded param
    cut_ratio = shrink_per / 100
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
            if ci == -1:
                continue # remove outliers
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            n = min(top_n, int(np.ceil(x_.size * top_per)))
            sort_table = []
            for xi, yi in zip(x_, y_):
                v = M[xi, yi]
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
            
    else:
        for xi, yi in zip(x, y):
            for i in range(-buff, buff+1):
                for j in range(-buff, buff+1):
                    candi.add((xi+i, yi+j))

    return c, candi

def select_intra_candidate(clr, chroms, Ed, k=100, q_thre=0.01, minv=1,
                           min_cluster_size=3, min_samples=3, shrink_per=15,
                           top_per=10, top_n=10, buff=2, nproc=4, highres=False):

    queue = []
    for c in chroms:
        queue.append((clr, c, Ed, k, q_thre, minv, min_cluster_size,
                      min_samples, shrink_per, top_per, top_n, buff,
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

def select_inter_core(clr, c1, c2, windows, min_per, q_thre=0.01,
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
    candi = set()
    top_per = top_per / 100
    max_cluster_size = 100 # hard-coded param
    cut_ratio = shrink_per / 100
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
            if ci == -1:
                continue # remove outliers
            mask = clusterer.labels_ == ci
            x_, y_ = x[mask], y[mask]
            n = min(top_n, int(np.ceil(x_.size * top_per)))
            sort_table = []
            for xi, yi in zip(x_, y_):
                v = M[xi, yi]
                sort_table.append((v, xi, yi))
            
            sort_table.sort(reverse=True)
            for _, xi, yi in sort_table[:n]:
                for i in range(-buff, buff+1):
                    for j in range(-buff, buff+1):
                        candi.add((xi+i, yi+j))
    else:
        for xi, yi in candidates_pool:
            for i in range(-buff, buff+1):
                for j in range(-buff, buff+1):
                    candi.add((xi+i, yi+j))
    
    return c1, c2, candi

def select_inter_candidate(clr, chroms, windows=[3,4,5], min_per=50,
                           q_thre=0.01, min_cluster_size=3, min_samples=3,
                           shrink_per=15, top_per=10, top_n=10, buff=2, nproc=4):

    queue = []
    for i in range(len(chroms)-1):
        for j in range(i+1, len(chroms)):
            queue.append((clr, chroms[i], chroms[j], windows, min_per, q_thre,
                          min_cluster_size, min_samples, shrink_per, top_per,
                          top_n, buff))
    
    results = Parallel(n_jobs=nproc)(delayed(select_inter_core)(*i) for i in queue)
    bychrom = {}
    for c1, c2, candi in results:
        if len(candi):
            bychrom[(c1, c2)] = candi
    
    return bychrom

def cross_resolution_mapping(by_res):

    mappable = {}
    resolutions = sorted(by_res)
    for i in range(len(resolutions)-1):
        for j in range(i+1, len(resolutions)):
            tr = resolutions[i]
            qr = resolutions[j]
            mappable[(tr, qr)] = defaultdict(set)
            mappable[(qr, tr)] = defaultdict(set)
            for k in by_res[tr]:
                if k in by_res[qr]:
                    for tx, ty in by_res[tr][k]:
                        s_l = range(tx*tr//qr, int(np.ceil((tx+1)*tr/qr)))
                        e_l = range(ty*tr//qr, int(np.ceil((ty+1)*tr/qr)))
                        for x in s_l:
                            for y in e_l:
                                if (x, y) in by_res[qr][k]:
                                    mappable[(tr, qr)][k].add((tx, ty))
                                    mappable[(qr, tr)][k].add((x, y))
    
    return mappable

def cross_resolution_support(by_res, level=2, intra=True):

    mapping_table = cross_resolution_mapping(by_res)
    new = {}
    for tr in sorted(by_res):
        new[tr] = defaultdict(set)
        for k in by_res[tr]:
            for tx, ty in by_res[tr][k]:
                valid = 0
                support = []
                if not intra:
                    for qr in by_res:
                        if qr == tr:
                            continue

                        valid += 1
                        if not k in mapping_table[(tr, qr)]:
                            support.append(False)
                        else:
                            if (tx, ty) in mapping_table[(tr, qr)][k]:
                                support.append(True)
                            else:
                                support.append(False)
                else:
                    for qr in by_res:
                        if qr == tr:
                            continue

                        dis = (ty - tx) * tr
                        if dis < 10 * qr:
                            continue

                        valid += 1
                        if not k in mapping_table[(tr, qr)]:
                            support.append(False)
                        else:
                            if (tx, ty) in mapping_table[(tr, qr)][k]:
                                support.append(True)
                            else:
                                support.append(False)
                
                if valid == 0:
                    new[tr][k].add((tx, ty))
                elif valid <= level:
                    if sum(support) == valid:
                        new[tr][k].add((tx, ty))
                else:
                    if sum(support) >= level:
                        new[tr][k].add((tx, ty))
    
    return new
