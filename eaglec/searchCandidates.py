import logging
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import defaultdict
from joblib import Parallel, delayed

log = logging.getLogger(__name__)

def select_intra_core(clr, c, Ed, k, q_thre, minv, highres=False):

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
        
    x_collect, y_collect, v_collect = x[mask], y[mask], v[mask]
    Poiss = stats.poisson(evalue)
    pvalues = Poiss.sf(v_collect)
    # check for short-range contacts
    idx = np.arange(M.shape[0])
    for i in range(10, k):
        diag = M.diagonal(i)
        xi = idx[:-i][diag>minv]
        yi = idx[i:][diag>minv]
        diag = diag[diag>minv]
        if diag.size > 0:
            x_collect = np.r_[x_collect, xi]
            y_collect = np.r_[y_collect, yi]
            Poiss = stats.poisson(Ed[i])
            pvalues = np.r_[pvalues, Poiss.sf(diag)]

    qvalues = multipletests(pvalues.ravel(), method='fdr_bh')[1]
    mask = qvalues < q_thre
    x_collect, y_collect = x_collect[mask], y_collect[mask]
    candi = set()
    for x, y in zip(x_collect, y_collect):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                candi.add((x+i, y+j))

    return c, candi

def select_intra_candidate(clr, chroms, Ed, k=20, q_thre=0.01, minv=1, nproc=4, highres=False):

    queue = []
    for c in chroms:
        queue.append((clr, c, Ed, k, q_thre, minv, highres))
    
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

def select_inter_core(clr, c1, c2, windows, min_per, less_stringent, q_thre):

    M = clr.matrix(balance=False, sparse=True).fetch(c1, c2)
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
    
    if not less_stringent:
        # contacts in each row/column are treated differently
        global_avg = v.mean()
        # column
        M = M.tocsc()
        indices, indptr = M.indices, M.indptr
        pvalues = []
        x_collect = []
        y_collect = []
        for i in range(M.shape[1]):
            col = M[indices[indptr[i]:indptr[i+1]], i].toarray().ravel()
            if col.size > 10:
                avg = col.mean()
            else:
                avg = global_avg

            Poiss = stats.poisson(avg)
            tmp_p = Poiss.sf(col)
            tmp_x = indices[indptr[i]:indptr[i+1]]
            tmp_y = np.ones_like(tmp_x) * i
            pvalues.append(tmp_p)
            x_collect.append(tmp_x)
            y_collect.append(tmp_y)
        
        pvalues = np.concatenate(pvalues)
        x_collect = np.concatenate(x_collect)
        y_collect = np.concatenate(y_collect)
        qvalues = multipletests(pvalues, method='fdr_bh')[1]
        mask = qvalues < q_thre
        x_collect, y_collect = x_collect[mask], y_collect[mask]
        coords = set(zip(x_collect, y_collect))
        candidates_pool.append(coords)

        # rows
        M = M.tocsr()
        indices, indptr = M.indices, M.indptr
        pvalues = []
        x_collect = []
        y_collect = []
        for i in range(M.shape[0]):
            row = M[i, indices[indptr[i]:indptr[i+1]]].toarray().ravel()
            if row.size > 10:
                avg = row.mean()
            else:
                avg = global_avg

            Poiss = stats.poisson(avg)
            tmp_p = Poiss.sf(row)
            tmp_y = indices[indptr[i]:indptr[i+1]]
            tmp_x = np.ones_like(tmp_y) * i
            pvalues.append(tmp_p)
            x_collect.append(tmp_x)
            y_collect.append(tmp_y)
        
        pvalues = np.concatenate(pvalues)
        x_collect = np.concatenate(x_collect)
        y_collect = np.concatenate(y_collect)
        qvalues = multipletests(pvalues, method='fdr_bh')[1]
        mask = qvalues < q_thre
        x_collect, y_collect = x_collect[mask], y_collect[mask]
        coords = set(zip(x_collect, y_collect))
        candidates_pool.append(coords)

    candi = set()
    for x, y in filter_candidates(candidates_pool):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                candi.add((x+i, y+j))

    return c1, c2, candi

def select_inter_candidate(clr, chroms, windows=[3,4,5], min_per=50,
                           less_stringent=False, q_thre=0.01, nproc=4):

    queue = []
    for i in range(len(chroms)-1):
        for j in range(i+1, len(chroms)):
            queue.append((clr, chroms[i], chroms[j], windows, min_per, less_stringent, q_thre))
    
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
