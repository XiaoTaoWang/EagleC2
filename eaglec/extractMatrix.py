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

def collect_images_core(mcool, res, c1, c2, coords, balance, exp, w, cachefolder):

    uri = '{0}::resolutions/{1}'.format(mcool, res)
    clr = cooler.Cooler(uri)
    Matrix = clr.matrix(balance=balance, sparse=True).fetch(c1, c2).tocsr()

    coords = np.r_[list(coords)]
    xi, yi = coords[:,0], coords[:,1]
    # chromosome boundary check
    mask = (xi - w >= 0) & (xi + w + 1 <= Matrix.shape[0]) & \
           (yi - w >= 0) & (yi + w + 1 <= Matrix.shape[1])
    xi, yi = xi[mask], yi[mask]

    batch_size = 10000
    count = 0
    # extract and normalize submatrices surrounding the input coordinates
    if xi.size > 0:
        seed = np.arange(-w, w+1)
        delta = np.tile(seed, (seed.size, 1))
        for t in range(0, xi.size, batch_size):
            data = []
            txi = xi[t:t+batch_size]
            tyi = yi[t:t+batch_size]
            xxx = txi.reshape((txi.size, 1, 1)) + delta.T
            yyy = tyi.reshape((tyi.size, 1, 1)) + delta
            v = np.array(Matrix[xxx.ravel(), yyy.ravel()]).ravel()
            vvv = v.reshape((txi.size, seed.size, seed.size))
            for i in range(txi.size):
                x = txi[i]
                y = tyi[i]
                window = vvv[i]
                window[np.isnan(window)] = 0
                    
                if not check_sparsity(window):
                    continue
                
                if c1 == c2:
                    window = distance_normaize_core(window, exp, x, y, w)
                
                window = image_normalize(window)

                data.append((window, (c1, x, c2, y, res)))
            
            outfil = os.path.join(cachefolder, 'collect.{0}_{1}.{2}.pkl'.format(c1, c2, t))
            joblib.dump(data, outfil, compress=('xz', 3))
            count += len(data)
    
    return count


def collect_images(mcool, by_res, expected_values, balance, cachefolder,
                   w=15, nproc=8):

    queue = []
    for res in by_res:
        for c1, c2 in by_res[res]:
            if c1 == c2:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, expected_values[res], w, cachefolder))
            else:
                queue.append((mcool, res, c1, c2, by_res[res][(c1, c2)],
                              balance, None, w, cachefolder))
    
    results = Parallel(n_jobs=nproc)(delayed(collect_images_core)(*i) for i in queue)
    total_n = 0
    for collect_n in results:
        total_n += collect_n
    
    return total_n
