import cooler, os, logging, joblib
import numpy as np
from eaglec.utilities import distance_normaize_core, image_normalize, entropy
from joblib import Parallel, delayed

log = logging.getLogger(__name__)

def check_sparsity(M):

    sub = M[5:-5, 5:-5] # 21 x 21
    nonzero = sub[sub.nonzero()]
    if nonzero.size < 10:
        return False
    else:
        return True

def collect_images_core(mcool, res, c1, c2, coords, balance, exp, w,
                        entropy_cutoff, cachefolder):

    uri = '{0}::resolutions/{1}'.format(mcool, res)
    clr = cooler.Cooler(uri)
    Matrix = clr.matrix(balance=balance, sparse=True).fetch(c1, c2).tocsr()

    coords = np.r_[list(coords)]
    xi, yi = coords[:,0], coords[:,1]
    # full window
    # chromosome boundary check
    mask_full = (xi - w >= 0) & (xi + w + 1 <= Matrix.shape[0]) & \
                (yi - w >= 0) & (yi + w + 1 <= Matrix.shape[1])
    x_full, y_full = xi[mask_full], yi[mask_full]

    batch_size = 10000
    count = 0
    # extract and normalize submatrices surrounding the input coordinates
    if x_full.size > 0:
        seed = np.arange(-w, w+1)
        delta = np.tile(seed, (seed.size, 1))
        for t in range(0, x_full.size, batch_size):
            data = []
            txi = x_full[t:t+batch_size]
            tyi = y_full[t:t+batch_size]
            xxx = txi.reshape((txi.size, 1, 1)) + delta.T
            yyy = tyi.reshape((tyi.size, 1, 1)) + delta
            v = np.array(Matrix[xxx.ravel(), yyy.ravel()]).ravel()
            vvv = v.reshape((txi.size, seed.size, seed.size))
            for i in range(txi.size):
                x = txi[i]
                y = tyi[i]
                window = vvv[i].astype(exp.dtype)
                window[np.isnan(window)] = 0
                
                if not check_sparsity(window):
                    continue
                
                if c1 == c2:
                    window = distance_normaize_core(window, exp, x, y, w)
                
                if entropy_cutoff < 1:
                    score = entropy(window, 11, 4)
                    if score > entropy_cutoff:
                        continue

                window = image_normalize(window)
                data.append((window, (c1, x, c2, y, res)))
            
            if len(data) > 0:
                count += len(data)
                outfil = os.path.join(cachefolder, 'collect_full.{0}_{1}_{2}.{3}.pkl'.format(c1, c2, res, t))
                joblib.dump(data, outfil, compress=('xz', 3))
    
    # half window, in case the breakpoints are located near the chromosome boundary
    w = w // 2
    xi = xi[np.logical_not(mask_full)]
    yi = yi[np.logical_not(mask_full)]
    mask_half = (xi - w >= 0) & (xi + w + 1 <= Matrix.shape[0]) & \
                (yi - w >= 0) & (yi + w + 1 <= Matrix.shape[1])
    x_half, y_half = xi[mask_half], yi[mask_half]
    if x_half.size > 0:
        seed = np.arange(-w, w+1)
        delta = np.tile(seed, (seed.size, 1))
        for t in range(0, x_half.size, batch_size):
            data = []
            txi = x_half[t:t+batch_size]
            tyi = y_half[t:t+batch_size]
            xxx = txi.reshape((txi.size, 1, 1)) + delta.T
            yyy = tyi.reshape((tyi.size, 1, 1)) + delta
            v = np.array(Matrix[xxx.ravel(), yyy.ravel()]).ravel()
            vvv = v.reshape((txi.size, seed.size, seed.size))
            for i in range(txi.size):
                x = txi[i]
                y = tyi[i]
                window = vvv[i].astype(exp.dtype)
                window[np.isnan(window)] = 0

                nonzero = window[window.nonzero()]   
                if nonzero.size < 10:
                    continue
                
                if c1 == c2:
                    window = distance_normaize_core(window, exp, x, y, w)
                
                if entropy_cutoff < 1:
                    score = entropy(window, 3, 4)
                    if score > entropy_cutoff:
                        continue

                window = image_normalize(window)
                window_full = np.random.random((31, 31)) * window[window>0].min()
                window_full[8:23, 8:23] = window
                data.append((window_full, (c1, x, c2, y, res)))
            
            if len(data) > 0:
                count += len(data)
                outfil = os.path.join(cachefolder, 'collect_half.{0}_{1}_{2}.{3}.pkl'.format(c1, c2, res, t))
                joblib.dump(data, outfil, compress=('xz', 3))
    
    return count


def collect_images(mcool, by_res, expected_values, balance, cachefolder,
                   w=15, entropy_cutoff=0.9, nproc=8):

    queue = []
    for res in by_res:
        for c1, c2 in by_res[res]:
            if c1 == c2:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, expected_values[res][c1], w, entropy_cutoff,
                              cachefolder))
            else:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, expected_values[res][c1], w, entropy_cutoff, cachefolder))
    
    results = Parallel(n_jobs=nproc)(delayed(collect_images_core)(*i) for i in queue)
    total_n = 0
    for collect_n in results:
        total_n += collect_n
    
    return total_n
