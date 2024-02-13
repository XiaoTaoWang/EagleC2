import joblib, cooler, os, logging
import numpy as np
from eaglec.utilities import distance_normaize_core, image_normalize
from joblib import Parallel, delayed

log = logging.getLogger(__name__)

def check_sparsity(M):

    sub = M[5:-5, 5:-5] # 21 x 21
    nonzero = sub[sub.nonzero()]
    if nonzero.size < 10:
        return False
    else:
        return True

def collect_images_core(mcool, res, c1, c2, coords, balance, exp, w):

    uri = '{0}::resolutions/{1}'.format(mcool, res)
    clr = cooler.Cooler(uri)
    Matrix = clr.matrix(balance=balance, sparse=True).fetch(c1, c2).tocsr()

    coords = np.r_[list(coords)]
    xi, yi = coords[:,0], coords[:,1]
    # chromosome boundary check
    mask = (xi - w >= 0) & (xi + w + 1 <= Matrix.shape[0]) & \
           (yi - w >= 0) & (yi + w + 1 <= Matrix.shape[1])
    xi, yi = xi[mask], yi[mask]

    # extract and normalize submatrices surrounding the input coordinates
    data = []
    if xi.size > 0:
        seed = np.arange(-w, w+1)
        delta = np.tile(seed, (seed.size, 1))
        xxx = xi.reshape((xi.size, 1, 1)) + delta.T
        yyy = yi.reshape((yi.size, 1, 1)) + delta
        v = np.array(Matrix[xxx.ravel(), yyy.ravel()]).ravel()
        vvv = v.reshape((xi.size, seed.size, seed.size))
        for i in range(xi.size):
            x = xi[i]
            y = yi[i]
            window = vvv[i]
            window[np.isnan(window)] = 0
                
            if not check_sparsity(window):
                continue
            
            if c1 == c2:
                window = distance_normaize_core(window, exp, x, y, w)
            
            window = image_normalize(window)

            data.append((window, (c1, x, c2, y, res)))
    
    return data

def collect_images(mcool, by_res, expected_values, balance, data_collect,
                   cachefolder, fidx=1, maxn=100000, w=15, nproc=8):

    queue = []
    for res in by_res:
        for c1, c2 in by_res[res]:
            if c1 == c2:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, expected_values[res], w))
            else:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, None, w))
    
    results = Parallel(n_jobs=nproc)(delayed(collect_images_core)(*i) for i in queue)
    for extract in results:
        for item in extract:
            data_collect.append(item)
            if len(data_collect) == maxn:
                outfil = os.path.join(cachefolder, 'collect_{0}.pkl'.format(fidx))
                joblib.dump(data_collect, outfil, compress=('xz', 3))
                log.info('Collected {0} images'.format(fidx*maxn))
                fidx += 1
                data_collect = []
    
    return data_collect, fidx
