import cooler, logging
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import dbscan

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


