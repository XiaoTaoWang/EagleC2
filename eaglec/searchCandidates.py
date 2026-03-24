import tensorflow as tf
tf.config.optimizer.set_jit(False)
import logging, cooler, os, joblib
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from eaglec.utilities import image_normalize

log = logging.getLogger(__name__)

@tf.function(reduce_retracing=True)
def local_minmax_normalize_2d(x2d, k=21, eps=1e-6):
    """
    x2d: tf.Tensor (H, W), float32
    returns: tf.Tensor (H, W) normalized to ~[0,1] by local k×k min/max
    """
    x = tf.cast(x2d, tf.float32)
    x4 = x[None, :, :, None]  # (1,H,W,1)

    # local max
    local_max = tf.nn.max_pool2d(x4, ksize=k, strides=1, padding="SAME")[0, :, :, 0]
    # local min via max_pool on -x
    local_min = -tf.nn.max_pool2d(-x4, ksize=k, strides=1, padding="SAME")[0, :, :, 0]

    denom = tf.maximum(local_max - local_min, eps)
    y = (x - local_min) / denom
    return tf.clip_by_value(y, 0.0, 1.0)

@tf.function(reduce_retracing=True)
def fcn_sv_probability_map(base_fcn, x2d_norm, neg_index=6):
    """
    base_fcn: (1,H,W,1) -> (1,H,W,7 logits)
    x2d_norm: tf.Tensor (H,W) in [0,1]
    returns: tf.Tensor (H,W) p_sv
    """
    x4 = x2d_norm[None, :, :, None]              # (1,H,W,1)
    logits = base_fcn(x4, training=False)        # (1,H,W,7)
    probs = tf.nn.softmax(logits, axis=-1)       # (1,H,W,7)
    p_non = probs[..., neg_index]                # (1,H,W)
    p_sv = 1.0 - p_non
    return p_sv[0]                               # (H,W)

def distance_normalize_block(block_dense, exp, R0, C0):
    """
    block_dense: (h,w) dense float32
    exp: 1D array where exp[d] is expected at distance d
    R0, C0: absolute start indices of this block in the full matrix

    Returns: distance-normalized block (h,w) float32
    """
    h, w = block_dense.shape

    # absolute row/col indices
    rows = (R0 + np.arange(h))[:, None]          # (h,1)
    cols = (C0 + np.arange(w))[None, :]          # (1,w)

    D = np.abs(rows - cols).astype(np.int32)     # (h,w)

    # if any distance in this block exceeds exp length, skip distance normalization
    if D.max() >= exp.size:
        return block_dense

    denom = exp[D]  # (h,w)
    out = np.divide(block_dense, denom, out=block_dense.copy(),
                    where=(denom != 0)).astype(np.float32, copy=False)

    return out

def iter_csr_tiles(M, tile_size=2048, k=21, exp=None, upper_triangular_only=False):
    """
    Traverse a CSR sparse matrix in tiles.

    For each tile core [r0:r1, c0:c1], we extract a halo-extended block
    [R0:R1, C0:C1], apply local window min-max normalization (k×k), and
    then return only the normalized core region.

    Yields:
      (r0, r1, c0, c1, core_norm)   where core_norm is a dense float32 array
                                    with shape (r1-r0, c1-c0)
    """
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()

    n_rows, n_cols = M.shape
    halo = (k - 1) // 2  # for k=21 => 10

    if not exp is None:
        exp = np.asarray(exp, dtype=np.float32)

    for r0 in range(0, n_rows, tile_size):
        r1 = min(r0 + tile_size, n_rows)
        c_start = r0 if upper_triangular_only else 0
        for c0 in range(c_start, n_cols, tile_size):
            c1 = min(c0 + tile_size, n_cols)

            # Extra fast-skip for strict lower triangle (in case of edge alignment)
            if upper_triangular_only and c1 <= r0:
                continue

            # halo-extended coordinates
            R0 = max(0, r0 - halo)
            R1 = min(n_rows, r1 + halo)
            C0 = max(0, c0 - halo)
            C1 = min(n_cols, c1 + halo)

            block = M[R0:R1, C0:C1]
            if block.nnz == 0:
                continue

            block_dense = block.toarray().astype(np.float32, copy=False)
            block_dense = np.nan_to_num(block_dense, nan=0.0, posinf=0.0, neginf=0.0)
            if not exp is None:
                block_dense = distance_normalize_block(block_dense, exp, R0, C0)

            # local normalization on the halo-extended dense block
            block_tf = tf.convert_to_tensor(block_dense, dtype=tf.float32)
            block_norm = local_minmax_normalize_2d(block_tf, k=k).numpy()

            # crop back to core tile (relative indices inside halo block)
            rr0 = r0 - R0
            rr1 = rr0 + (r1 - r0)
            cc0 = c0 - C0
            cc1 = cc0 + (c1 - c0)

            core_norm = block_norm[rr0:rr1, cc0:cc1]
            yield (r0, r1, c0, c1, core_norm)

def iter_cooler_scan_candidates(cool_path, resolutions, chroms, expected_values,
                                balance, base_fcn, tile_size=2048, k=21, cutoff=0.3):
    
    candidates = {}
    count = 0
    for res in resolutions:
        clr = cooler.Cooler('{0}::resolutions/{1}'.format(cool_path, res))
        candidates[res] = defaultdict(list)
        # cis
        for chrom in chroms:
            log.info('  Scanning {0} at resolution {1} ...'.format(chrom, res))
            M = clr.matrix(balance=balance, sparse=True).fetch(chrom).tocsr()
            for r0, r1, c0, c1, core_norm in iter_csr_tiles(
                M,
                tile_size=tile_size,
                k=k,
                exp=expected_values[res][chrom],
                upper_triangular_only=True
            ):
                x2d = tf.convert_to_tensor(core_norm, dtype=tf.float32)
                p_sv_np = fcn_sv_probability_map(base_fcn, x2d).numpy()
                mask = p_sv_np > cutoff
                ii_list, jj_list = np.where(mask)
                for ii, jj in zip(ii_list, jj_list):
                    abs_i = r0 + ii
                    abs_j = c0 + jj
                    candidates[res][(chrom, chrom)].append((abs_i, abs_j, float(p_sv_np[ii, jj])))
                    count += 1

        # trans        
        for i in range(len(chroms)-1):
            for j in range(i+1, len(chroms)):
                chrom1, chrom2 = chroms[i], chroms[j]
                log.info('  Scanning {0} vs {1} at resolution {2} ...'.format(chrom1, chrom2, res))
                M = clr.matrix(balance=balance, sparse=True).fetch(chrom1, chrom2).tocsr()
                for r0, r1, c0, c1, core_norm in iter_csr_tiles(
                    M,
                    tile_size=tile_size,
                    k=k,
                    exp=None,
                    upper_triangular_only=False
                ):
                    x2d = tf.convert_to_tensor(core_norm, dtype=tf.float32)
                    p_sv_np = fcn_sv_probability_map(base_fcn, x2d).numpy()
                    mask = p_sv_np > cutoff
                    ii_list, jj_list = np.where(mask)
                    for ii, jj in zip(ii_list, jj_list):
                        abs_i = r0 + ii
                        abs_j = c0 + jj
                        candidates[res][(chrom1, chrom2)].append((abs_i, abs_j, float(p_sv_np[ii, jj])))
                        count += 1
        
    return candidates, count

def extract_centered_patch_from_matrix(M, center_i, center_j, radius=15, exp=None,
                                       pad_value=0.0):
    """
    Extract a fixed-size patch centered at (center_i, center_j) from full matrix M.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        Whole chromosome-wide or chromosome-pair matrix.
    center_i, center_j : int
        Absolute bin coordinates within M.
    radius : int
        Patch radius. radius=15 gives a 31x31 patch.
    exp : 1D np.ndarray or None
        Expected vector for cis matrices. None for trans.
    pad_value : float
        Value used when patch crosses chromosome boundary.

    Returns
    -------
    out : np.ndarray, shape (2*radius+1, 2*radius+1), dtype float32
    """
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()

    n_rows, n_cols = M.shape
    out_size = 2 * radius + 1

    r0 = max(0, center_i - radius)
    r1 = min(n_rows, center_i + radius + 1)
    c0 = max(0, center_j - radius)
    c1 = min(n_cols, center_j + radius + 1)

    block = M[r0:r1, c0:c1].toarray().astype(np.float32, copy=False)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    if not exp is None:
        block = distance_normalize_block(block, exp, r0, c0)

    block = image_normalize(block)

    out = np.full((out_size, out_size), pad_value, dtype=np.float32)

    rr0 = radius - (center_i - r0)
    cc0 = radius - (center_j - c0)
    out[rr0:rr0 + (r1 - r0), cc0:cc0 + (c1 - c0)] = block

    return out

def check_sparsity(patch, margin=5, min_nonzero=10):

    sub = patch[margin:-margin, margin:-margin]

    return np.count_nonzero(sub) >= min_nonzero

def collect_candidate_patches(cool_path, candidates, expected_values, out_dir,
                              balance, radius=15, chunk_size=10000):
    """
    Re-extract centered patches from full chromosome-wide / chromosome-pair matrices
    using candidates organized as:

        candidates[res][(chrom1, chrom2)] = [(abs_i, abs_j, score), ...]

    Parameters
    ----------
    cool_path : str
    candidates : dict
        Output of iter_cooler_scan_candidates.
    expected_values : dict
        expected_values[res][chrom] for cis normalization.
    out_dir : str
    balance : str
    radius : int
        radius=15 gives 31x31 patches.
    chunk_size : int

    Returns
    -------
    patch_count : int
        Number of patches collected.
    """
    collect_items = []
    patch_count = 0

    for res in sorted(candidates.keys()):
        log.info('Collecting patches at resolution {0} ...'.format(res))
        clr = cooler.Cooler('{0}::resolutions/{1}'.format(cool_path, res))

        # cache one matrix per chromosome pair to avoid repeated fetch
        for chrom_pair in candidates[res]:
            chrom1, chrom2 = chrom_pair
            M = clr.matrix(balance=balance, sparse=True).fetch(chrom1, chrom2).tocsr()
            exp = expected_values[res][chrom1] if chrom1 == chrom2 else None

            for abs_i, abs_j, score in candidates[res][chrom_pair]:
                if chrom1 == chrom2:
                    if abs_j - abs_i < 6:
                        continue
                
                patch = extract_centered_patch_from_matrix(
                    M,
                    center_i=abs_i,
                    center_j=abs_j,
                    radius=radius,
                    exp=exp
                )

                if not check_sparsity(patch):
                    continue

                collect_items.append((patch, (res, chrom1, abs_i, chrom2, abs_j, score)))
                patch_count += 1

                if len(collect_items) >= chunk_size:
                    outfil = os.path.join(out_dir, 'collect_items.{0}.pkl'.format(patch_count))
                    joblib.dump(collect_items, outfil, compress=('xz', 3))
                    collect_items = []

    if len(collect_items) > 0:
        outfil = os.path.join(out_dir, 'collect_items.{0}.pkl'.format(patch_count))
        joblib.dump(collect_items, outfil, compress=('xz', 3))

    return patch_count