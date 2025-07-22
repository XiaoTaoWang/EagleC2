import cooler, logging, joblib, os, eaglec, glob, math
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from numba import njit

log = logging.getLogger(__name__)

class SVblock(object):

    def __init__(self, clr, sv, exp, balance='sweight'):

        # exp: returned by calculate_expected
        c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6 = sv[:10]
        SV_labels = ['++', '+-', '-+', '--', '++/--', '+-/-+']
        prob = np.array([prob1, prob2, prob3, prob4, prob5, prob6])
        maxi = prob.argmax()
        strands = SV_labels[maxi].split('/')

        self.p1 = p1 // clr.binsize
        self.p2 = p2 // clr.binsize
        self.chromsize1 = clr.chromsizes[c1] // clr.binsize
        self.chromsize2 = clr.chromsizes[c2] // clr.binsize
        self.clr = clr
        self.strands = strands
        self.exp = exp
        self.balance = balance
        self.c1 = c1
        self.c2 = c2
        

    def get_matrices(self, strand, w):

        clr, c1, c2, x, y = self.clr, self.c1, self.c2, self.p1, self.p2
        Matrix = clr.matrix(balance=self.balance, sparse=True).fetch(c1, c2).tocsr()
        M1 = Matrix[x-w:x+w+1, y-w:y+w+1].toarray()
        M1[np.isnan(M1)] = 0
        if c1 != c2:
            M2 = M1.copy()
        else:
            M1 = M1.astype(self.exp[c1].dtype)
            M2 = distance_normaize_core(M1, self.exp[c1], x, y, w)    

        if strand == '++':
            M1 = M1[:(w+1), :(w+1)]
            M2 = M2[:(w+1), :(w+1)]
            M1 = M1[:,::-1]
            M2 = M2[:,::-1]
        elif strand == '+-':
            M1 = M1[:(w+1), w:]
            M2 = M2[:(w+1), w:]
        elif strand == '-+':
            M1 = M1[w:, :(w+1)]
            M2 = M2[w:, :(w+1)]
            M1 = M1[::-1,::-1]
            M2 = M2[::-1,::-1]
        else:
            M1 = M1[w:, w:]
            M2 = M2[w:, w:]
            M1 = M1[::-1,:]
            M2 = M2[::-1,:]
        
        return M1, M2
    
    def detect_bounds(self, M):

        from sklearn.decomposition import PCA
        from scipy.ndimage import gaussian_filter

        u_i = M.shape[0]//2
        d_i = M.shape[1]//2
        u_scores = {k:-1 for k in range(M.shape[0])}
        d_scores = {k:-1 for k in range(M.shape[1])}
        # locate the upstream and downstream bound independently
        rowmask = M.sum(axis=1) != 0
        colmask = M.sum(axis=0) != 0
        if (rowmask.sum() >= 10) and (colmask.sum() >= 10):
            # row, upstream
            new = M[rowmask][:,colmask]
            corr = gaussian_filter(np.corrcoef(new, rowvar=True), sigma=1)
            pca = PCA(n_components=3, whiten=True)
            pc1_row = pca.fit_transform(corr)[:,0]
            t_i, t_scores = self.locate(pc1_row, rowmask)
            if not t_i is None:
                u_i = t_i
            if len(t_scores):
                for k in t_scores:
                    u_scores[k] = t_scores[k]

            # column, downstream
            corr = gaussian_filter(np.corrcoef(new, rowvar=False), sigma=1)
            pca = PCA(n_components=3, whiten=True)
            pc1_col = pca.fit_transform(corr)[:,0]
            t_i, t_scores = self.locate(pc1_col, colmask)
            if not t_i is None:
                d_i = t_i
            if len(t_scores):
                for k in t_scores:
                    d_scores[k] = t_scores[k]
        
        return u_i, d_i, u_scores, d_scores
    
    def locate(self, curve, mask, cutoff=2.58):

        from scipy.stats import median_abs_deviation

        coords_map = np.where(mask)[0]
        # identify points with significant value changes compared with the previous points
        diff = np.abs(np.diff(curve))
        mad = median_abs_deviation(diff)
        median = np.median(diff)
        z_scores = 0.6745 * (diff - median) / mad
        candidates = np.where(z_scores > cutoff)[0] + 1
        sort_table = sorted(zip(z_scores[candidates-1], candidates))
        sort_table.sort(reverse=True)
        D = dict(zip(coords_map[1:], z_scores))

        # identify stretches of points with the same trend of value change
        intervals = []
        diff = np.diff(curve)
        si = 0
        ei = 1
        for i in range(1, diff.size):
            if np.sign(diff[i]) == np.sign(diff[i-1]):
                ei += 1
                if i == (diff.size - 1):
                    intervals.append((si, ei))
            else:
                if (i < diff.size - 1) and (np.sign(diff[i+1]) == np.sign(diff[i-1])) and (np.abs(diff[i+1]) > np.abs(diff[i])*2):
                    ei += 1
                    diff[i] = -diff[i]
                else:
                    intervals.append((si, ei)) # (si, ei]
                    if i == (diff.size - 1):
                        intervals.append((ei, ei+1))
                    else:
                        si = ei
                        ei = si + 1
        
        # locate the most possible boundary
        bound = None
        for score, pos in sort_table:
            check = False
            for si, ei in intervals:
                if (si < pos <= ei) and (np.sign(curve[si]) != np.sign(curve[ei])) and (pos > 2) and (len(curve)-pos > 2):
                    check = True
                    break
            if check:
                bound = coords_map[pos]
                break

        return bound, D
    
    def check_distance_decay(self, M, min_block_width=3, N=10, dynamic_window_size=4,
                             min_point_num=10, rscore_cutoff=0.64):
        
        correlation = 0
        rowmask = M.sum(axis=1) != 0
        colmask = M.sum(axis=0) != 0
        if (rowmask.sum() < min_block_width) or (colmask.sum() < min_block_width):
            correlation = 0 # insufficient data
        else:
            x_arr = np.arange(0, M.shape[0]).reshape((M.shape[0], 1))
            y_arr = np.arange(M.shape[0]-1, M.shape[0]+M.shape[1]-1)
            valid = rowmask.reshape((M.shape[0], 1)) & colmask
            D = y_arr - x_arr
            maxdis = min(D.max(), self.exp[self.c1].size-1)
            diag_sums = np.zeros(maxdis+1)
            pixel_nums = np.zeros(maxdis+1)
            for i in range(maxdis+1):
                mask = (D == i) & valid
                diag = M[mask]
                if diag.size > 0:
                    diag_sums[i] = diag.sum()
                    pixel_nums[i] = diag.size
            
            Ed = {}
            for i in range(maxdis+1):
                for w in range(dynamic_window_size+1):
                    tmp_sums = diag_sums[max(i-w,0):i+w+1]
                    tmp_nums = pixel_nums[max(i-w,0):i+w+1]
                    n_count = sum(tmp_sums)
                    n_pixel = sum(tmp_nums)
                    if n_pixel > N:
                        Ed[i] = n_count / n_pixel
                        break
            
            if len(Ed) < min_point_num:
                correlation = 0 # insufficient data
            else:
                Xi = np.r_[sorted(Ed)]
                X = np.r_[[self.exp[self.c1][i] for i in sorted(Ed)]]
                Y = np.r_[[Ed[i] for i in sorted(Ed)]]
                warning, increasing_bool = check_increasing(Xi, Y)
                if increasing_bool:
                    correlation = -1
                else:
                    IR = IsotonicRegression(increasing=increasing_bool)
                    IR.fit(Xi, Y)
                    vi = np.where(np.diff(IR.predict(Xi)) < 0)[0]
                    if vi.size > 0:
                        si = min(vi[0], 5)
                    else:
                        si = 0
                    labels = []
                    step = max(1, (len(Ed) - si - min_point_num + 1) // 5)
                    for num in range(si+min_point_num, len(Ed)+1, step):
                        tX = X[si:num][:,np.newaxis]
                        tY = Y[si:num]
                        ln = LinearRegression().fit(tX, tY)
                        rscore = ln.score(tX, tY)
                        slope = ln.coef_[0]
                        if (rscore > rscore_cutoff) and (slope > 0):
                            labels.append(1)
                        else:
                            labels.append(-1)
                        
                        if tX.size > X.size * 0.5:
                            break
                    
                    if not len(labels):
                        correlation = 0
                    else:
                        if sum(labels) >= 0:
                            correlation = 1
                        else:
                            correlation = -1
        
        return correlation

def check_increasing(x, y):
    """Determine whether y is monotonically correlated with x.
    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.
    Parameters
    ----------
    x : array-like of shape (n_samples,)
            Training data.
    y : array-like of shape (n_samples,)
        Training target.
    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.
    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.
    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.
    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation
    """

    # Calculate Spearman rho estimate and set return accordingly.
    rho, _ = spearmanr(x, y)
    warning = False
    increasing_bool = rho >= 0

    # Run Fisher transform to get the rho CI, but handle rho=+/-1
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1. + rho) / (1. - rho))
        F_se = 1 / math.sqrt(len(x) - 3)

        # Use a 95% CI, i.e., +/-1.96 S.E.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)

        # Warn if the CI spans zero.
        if np.sign(rho_0) != np.sign(rho_1):
            warning = True

    return warning, increasing_bool

def find_matched_resolution(expected_values, res):

    res_list = list(expected_values.keys())
    diff = np.abs(res - np.r_[res_list])
    idx = np.argmin(diff)

    return res_list[idx]

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

def get_queue(cache_folder, maxn=100000, pattern='collect*.pkl'):

    if type(pattern) == list:
        files = pattern
    else:
        files = glob.glob(os.path.join(cache_folder, pattern))
        
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
    
