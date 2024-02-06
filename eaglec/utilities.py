import cooler, logging
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import dbscan
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

log = logging.getLogger(__name__)

def get_valid_cols(clr, c, balance):

    if balance:
        weights = clr.bins().fetch(c)[balance].values
        valid_cols = np.isfinite(weights) & (weights > 0)
    else:
        M = clr.matrix(balance=False, sparse=True).fetch(c).tocsr()
        marg = np.array(M.sum(axis=0)).ravel()
        logNzMarg = np.log(marg[marg>0])
        med_logNzMarg = np.median(logNzMarg)
        dev_logNzMarg = cooler.balance.mad(logNzMarg)
        cutoff = np.exp(med_logNzMarg - 30 * dev_logNzMarg)
        marg[marg<cutoff] = 0
        valid_cols = marg > 0
    
    return valid_cols

def calculate_expected_core(clr, c, balance, max_dis):

    M = clr.matrix(balance=balance, sparse=True).fetch(c).tocsr()
    valid_cols = get_valid_cols(clr, c, balance)
    n = M.shape[0]

    expected = {}
    maxdis = min(n-1, max_dis)
    for i in range(maxdis+1):
        if i == 0:
            valid = valid_cols
        else:
            valid = valid_cols[:-i] * valid_cols[i:]

        diag = M.diagonal(i)[valid]
        if diag.size > 0:
            expected[i] = [diag.sum(), diag.size]
    
    return expected

def calculate_expected(clr, chroms, balance, max_dis, nproc=4,
                       N=20, dynamic_window_size=10):

    queue = []
    res = clr.binsize
    for c in chroms:
        queue.append((clr, c, balance, max_dis))
    
    results = Parallel(n_jobs=nproc)(delayed(calculate_expected_core)(*i) for i in queue)
    diag_sums = []
    pixel_nums = []
    for i in range(max_dis+1):
        nume = 0
        denom = 0
        for extract in results:
            if i in extract:
                nume += extract[i][0]
                denom += extract[i][1]
        diag_sums.append(nume)
        pixel_nums.append(denom)
    
    Ed = {}
    for i in range(max_dis+1):
        for w in range(dynamic_window_size+1):
            tmp_sums = diag_sums[max(i-w,0):i+w+1]
            tmp_nums = pixel_nums[max(i-w,0):i+w+1]
            n_count = sum(tmp_sums)
            n_pixel = sum(tmp_nums)
            if n_pixel > N:
                Ed[i] = n_count / n_pixel
                break
    
    IR = IsotonicRegression(increasing=False, out_of_bounds='clip')
    IR.fit(sorted(Ed), [Ed[i] for i in sorted(Ed)])
    d = np.arange(max(Ed)+1)
    exp_arr = IR.predict(list(d))
    Ed = dict(zip(d, exp_arr))
    
    return Ed

def select_intra_core(clr, c, Ed, k, q_thre, minv, highres=False):

    M = clr.matrix(balance=False, sparse=True).fetch(c).tocsr()
    x, y = M.nonzero()
    v = M.data
    # check for long-range contacts
    evalue = Ed[k]
    dis = y - x
    if highres:
        mask = (dis >= k) & (dis <= max(Ed)) & (v > minv)
    else:
        mask = (dis >= k) & (v > minv)
        
    x_collect, y_collect, v_collect = x[mask], y[mask], v[mask]
    del x, y, v, dis

    Poiss = stats.poisson(evalue)
    pvalues = Poiss.sf(v_collect)
    # check for short-range contacts
    idx = np.arange(M.shape[0])
    for i in range(9, k):
        diag = M.diagonal(i)
        xi = idx[:-i][diag>minv]
        yi = idx[i:][diag>minv]
        diag = diag[diag>minv]
        x_collect = np.r_[x_collect, xi]
        y_collect = np.r_[y_collect, yi]
        Poiss = stats.poisson(Ed[i])
        pvalues = np.r_[pvalues, Poiss.sf(diag)]

    qvalues = multipletests(pvalues.ravel(), method='fdr_bh')[1]
    mask = qvalues < q_thre
    x_collect, y_collect = x_collect[mask], y_collect[mask]
    candi = set(zip(x_collect, y_collect))

    del M, x_collect, y_collect

    return c, candi

def select_intra_candidate(clr, chroms, Ed, k=20, q_thre=0.01, minv=1, nproc=4, highres=False):

    queue = []
    for c in chroms:
        queue.append((clr, c, Ed, k, q_thre, minv-1, highres))
    
    results = Parallel(n_jobs=nproc)(delayed(select_intra_core)(*i) for i in queue)
    bychrom = {}
    for c, candi in results:
        bychrom[c] = candi
    
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

def select_inter_core(clr, c1, c2, windows, min_per, q_thre):

    M = clr.matrix(balance=False, sparse=True).fetch(c1, c2)
    x, y = M.nonzero()
    v = M.data
    
    candidates_pool = []
    for w in windows:
        x_edge = generage_bin_edges(M.shape[0], w)
        y_edge = generage_bin_edges(M.shape[1], w)
        sum_v, x_edge, y_edge, bin_indices = stats.binned_statistic_2d(
            x, y, v, statistic='sum', bins=(x_edge, y_edge), expand_binnumbers=True
        )

        l_v = np.percentile(sum_v[sum_v>0], min_per)
        mask = sum_v > l_v
        if mask.sum() > 5000:
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
    
    candidates = filter_candidates(candidates_pool)

    return c1, c2, candidates

def select_inter_candidate(clr, chroms, windows=[3, 4, 5], min_per=75, q_thre=0.01, nproc=4):

    queue = []
    for i in range(len(chroms)-1):
        for j in range(i+1, len(chroms)):
            queue.append((clr, chroms[i], chroms[j], windows, min_per, q_thre))
    
    results = Parallel(n_jobs=nproc)(delayed(select_inter_core)(*i) for i in queue)
    bychrom = {}
    for c1, c2, candi in results:
        bychrom[(c1, c2)] = candi
    
    return bychrom

def check_in(x, y, tr, qr, pool):

    s_l = range(x*tr//qr, int(np.ceil((x+1)*tr/qr)))
    e_l = range(y*tr//qr, int(np.ceil((y+1)*tr/qr)))
    B = False
    for i in s_l:
        for j in e_l:
            if (i, j) in pool:
                B = True
                break
    
    return B

def cross_resolution_support(by_res, level=2, intra=True):

    new = {}
    for tr in sorted(by_res):
        new[tr] = {}
        for k in by_res[tr]:
            new[tr][k] = set()
            for tx, ty in by_res[tr][k]:
                valid = 0
                support = []
                if not intra:
                    for qr in by_res:
                        if qr == tr:
                            continue

                        valid += 1
                        if not k in by_res[qr]:
                            support.append(False)
                        else:
                            support.append(check_in(tx, ty, tr, qr, by_res[qr][k]))
                else:
                    for qr in by_res:
                        if qr == tr:
                            continue

                        dis = (ty - tx) * tr
                        if dis < 10 * qr:
                            continue

                        valid += 1
                        if not k in by_res[qr]:
                            support.append(False)
                        else:
                            support.append(check_in(tx, ty, tr, qr, by_res[qr][k]))
                
                if valid == 0:
                    new[tr][k].add((tx, ty))
                elif valid <= level:
                    if sum(support) == valid:
                        new[tr][k].add((tx, ty))
                else:
                    if sum(support) >= level:
                        new[tr][k].add((tx, ty))
    
    return new



