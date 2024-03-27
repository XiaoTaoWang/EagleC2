import cooler, logging, joblib, os, eaglec, glob
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from numba import njit

log = logging.getLogger(__name__)

def get_queue(cache_folder, maxn=100000):

    files = glob.glob(os.path.join(cache_folder, 'collect.*.pkl'))
    data_collect = []
    for f in files:
        extract = joblib.load(f)
        for item in extract:
            data_collect.append(item)
            if len(data_collect) == maxn:
                yield data_collect
                data_collect = []
    
    if len(data_collect):
        yield data_collect


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
    
    return exp_arr

def load_gap(clr, chroms, ref_genome='hg38', balance='weight'):

    gaps = {}
    if ref_genome in ['hg19', 'hg38', 'chm13']:
        folder = os.path.join(os.path.split(eaglec.__file__)[0], 'data')
        if clr.binsize <= 10000:
            ref_gaps = joblib.load(os.path.join(folder, '{0}.gap-mask.10k.pkl'.format(ref_genome)))
        elif 10000 < clr.binsize <= 50000:
            ref_gaps = joblib.load(os.path.join(folder, '{0}.gap-mask.50k.pkl'.format(ref_genome)))
        else:
            ref_gaps = joblib.load(os.path.join(folder, '{0}.gap-mask.500k.pkl'.format(ref_genome)))

        for c in chroms:
            valid_bins = get_valid_cols(clr, c, balance)
            valid_idx = np.where(valid_bins)[0]
            chromlabel = 'chr'+c.lstrip('chr')
            gaps[c] = np.zeros(len(clr.bins().fetch(c)), dtype=bool)
            if chromlabel in ref_gaps:
                for i in range(len(gaps[c])):
                    if clr.binsize <= 500000:
                        if clr.binsize <= 10000:
                            ref_i = i * clr.binsize // 10000
                        elif 10000 < clr.binsize <= 50000:
                            ref_i = i * clr.binsize // 50000
                        else:
                            ref_i = i * clr.binsize // 500000
                        if ref_gaps[chromlabel][ref_i]:
                            gaps[c][i] = True

                gaps[c][valid_idx] = False
    else:
        for c in chroms:
            gaps[c] = np.zeros(len(clr.bins().fetch(c)), dtype=bool)

    return gaps
    
@njit
def distance_normaize_core(sub, exp, x, y, w):

    # calculate x and y indices
    x_arr = np.arange(x-w, x+w+1).reshape((2*w+1, 1))
    y_arr = np.arange(y-w, y+w+1)

    D = y_arr - x_arr
    D = np.abs(D)
    min_dis = D.min()
    max_dis = D.max()
    if max_dis >= exp.size:
        return sub
    else:
        exp_sub = np.zeros(sub.shape)
        for d in range(min_dis, max_dis+1):
            xi, yi = np.where(D==d)
            for i, j in zip(xi, yi):
                exp_sub[i, j] = exp[d]
            
        normed = sub / exp_sub

        return normed
    
@njit
def image_normalize(arr_2d):

    arr_2d = (arr_2d - arr_2d.min()) / (arr_2d.max() - arr_2d.min()) # value range: [0,1]

    return arr_2d