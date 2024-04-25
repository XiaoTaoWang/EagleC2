import cooler, logging, joblib, os, eaglec, glob
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from numba import njit

log = logging.getLogger(__name__)

def dict2list(D, res):

    L = []
    for sv in D:
        for c1, c2 in D[sv]:
            for p1, p2 in D[sv][(c1, c2)]:
                line = (c1, p1*res, c2, p2*res) + tuple(D[sv][(c1, c2)][(p1, p2)]) + (res, res)
                L.append(line)
    
    return L

def list2dict(L, res):

    D = {}
    SV_labels = ['++', '+-', '-+', '--', '++/--', '+-/-+']
    for line in L:
        c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6 = line[:10]
        p1 = p1 // res
        p2 = p2 // res
        prob = np.array([prob1, prob2, prob3, prob4, prob5, prob6])
        maxi = prob.argmax()
        sv = SV_labels[maxi]
        if not sv in D:
            D[sv] = {}
        
        if not (c1, c2) in D[sv]:
            D[sv][(c1, c2)] = {}
        
        D[sv][(c1, c2)][(p1, p2)] = prob
    
    return D

def get_queue(cache_folder, maxn=100000):

    files = glob.glob(os.path.join(cache_folder, 'collect*.pkl'))
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
    
    return c, expected

def calculate_expected(clr, chroms, balance, max_dis, nproc=4,
                       N=50, dynamic_window_size=2):

    res = clr.binsize
    queue = []
    diag_sums = {}
    pixel_nums = {}
    for c in chroms:
        queue.append((clr, c, balance, max_dis))
        diag_sums[c] = np.zeros(max_dis+1)
        pixel_nums[c] = np.zeros(max_dis+1)
    diag_sums['genome'] = np.zeros(max_dis+1)
    pixel_nums['genome'] = np.zeros(max_dis+1)

    results = Parallel(n_jobs=nproc)(delayed(calculate_expected_core)(*i) for i in queue)
    for i in range(max_dis+1):
        nume = 0 # genome-wide aggregation
        denom = 0
        for c, extract in results:
            if i in extract:
                nume += extract[i][0]
                denom += extract[i][1]
                diag_sums[c][i] = extract[i][0]
                pixel_nums[c][i] = extract[i][1]
        diag_sums['genome'][i] = nume
        pixel_nums['genome'][i] = denom
    
    Ed = {}
    for c in diag_sums:
        tmp = {}
        for i in range(max_dis+1):
            for w in range(dynamic_window_size+1):
                tmp_sums = diag_sums[c][max(i-w,0):i+w+1]
                tmp_nums = pixel_nums[c][max(i-w,0):i+w+1]
                n_count = sum(tmp_sums)
                n_pixel = sum(tmp_nums)
                if n_pixel > N:
                    tmp[i] = n_count / n_pixel
                    break
        Ed[c] = tmp
    
    exp_bychrom = {}
    for c in Ed:
        if len(Ed[c]) < len(Ed['genome'])*0.9:
            Ed[c] = Ed['genome'] 
    
        IR = IsotonicRegression(increasing=False, out_of_bounds='clip')
        IR.fit(sorted(Ed[c]), [Ed[c][i] for i in sorted(Ed[c])])
        d = np.arange(max_dis+1)
        exp_bychrom[c] = IR.predict(list(d))
        
    return exp_bychrom

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
def local_background(sub, exp, x, y, w):

    # calculate x and y indices
    x_arr = np.arange(x-w, x+w+1).reshape((2*w+1, 1))
    y_arr = np.arange(y-w, y+w+1)

    D = y_arr - x_arr
    D = np.abs(D)
    min_dis = D.min()
    max_dis = D.max()
    if max_dis >= exp.size:
        xi, yi = np.where(sub>0)
        nonzeros = np.zeros(xi.size)
        for i in range(xi.size):
            nonzeros[i] = sub[xi[i], yi[i]]
        E_ = nonzeros.mean()
    else:
        exp_sub = np.zeros(sub.shape)
        for d in range(min_dis, max_dis+1):
            xi, yi = np.where(D==d)
            for i, j in zip(xi, yi):
                exp_sub[i, j] = exp[d]
        
        xi, yi = np.where(sub>0)
        sub_ = np.zeros(xi.size)
        exp_ = np.zeros(xi.size)
        for i in range(xi.size):
            sub_[i] = sub[xi[i], yi[i]]
            exp_[i] = exp_sub[xi[i], yi[i]]

        E_ = sub_.sum() / exp_.sum() * exp[y-x]
    
    return E_

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

@njit
def entropy(M, si, w):

    sub1 = M[si:(si+w), si:(si+w)].sum() # ++
    sub2 = M[si:(si+w), (si+w+1):(si+2*w+1)].sum() # +-
    sub3 = M[(si+w+1):(si+2*w+1), si:(si+w)].sum() # -+
    sub4 = M[(si+w+1):(si+2*w+1), (si+w+1):(si+2*w+1)].sum() # --
    if sub1 == 0:
        sub1 = 1e-10
    if sub2 == 0:
        sub2 = 1e-10
    if sub3 == 0:
        sub3 = 1e-10
    if sub4 == 0:
        sub4 = 1e-10

    score_table = []
    total = sub1 + sub2
    score1 = 0
    for sub in [sub1, sub2]:
        p = sub / total
        score1 += (p * np.log2(p))
    score1 = -score1
    score_table.append((total, score1))

    total = sub1 + sub3
    score2 = 0
    for sub in [sub1, sub3]:
        p = sub / total
        score2 += (p * np.log2(p))
    score2 = -score2
    score_table.append((total, score2))

    total = sub3 + sub4
    score3 = 0
    for sub in [sub3, sub4]:
        p = sub / total
        score3 += (p * np.log2(p))
    score3 = -score3
    score_table.append((total, score3))

    total = sub2 + sub4
    score4 = 0
    for sub in [sub2, sub4]:
        p = sub / total
        score4 += (p * np.log2(p))
    score4 = -score4
    score_table.append((total, score4))
    score_table.sort()

    score = score_table[-1][1]

    return score
    
