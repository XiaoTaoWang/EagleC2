import tensorflow as tf
tf.config.optimizer.set_jit(False)
import logging, cooler, os, joblib, operator
import numpy as np
import scipy.sparse as sp
from numba import njit
from collections import defaultdict
from eaglec.utilities import image_normalize

log = logging.getLogger(__name__)


@njit(cache=True)
def _halo_local_pass(ptr, col, i, j, wr0, wr1, wc0, wc1, scratch):
    """Check 21x21 windows containing a 2x2 and centered in one core tile."""
    r0 = max(wr0, i - 19)
    r1 = min(i, wr1 - 21)
    c0 = max(wc0, j - 19)
    c1 = min(j, wc1 - 21)
    if r0 > r1 or c0 > c1:
        return False

    h = r1 + 21 - r0
    w = c1 + 21 - c0
    for row in range(h + 1):
        for column in range(w + 1):
            scratch[row, column] = 0

    total = 0
    for row in range(h):
        lo = ptr[r0 + row]
        hi = ptr[r0 + row + 1]
        end = hi
        while lo < hi:
            mid = (lo + hi) // 2
            if col[mid] < c0:
                lo = mid + 1
            else:
                hi = mid
        while lo < end and col[lo] < c0 + w:
            scratch[row + 1, col[lo] - c0 + 1] = 1
            total += 1
            lo += 1

    if total < 10:
        return False

    for row in range(1, h + 1):
        row_sum = 0
        for column in range(1, w + 1):
            row_sum += scratch[row, column]
            scratch[row, column] = scratch[row - 1, column] + row_sum

    for row in range(r1 - r0 + 1):
        for column in range(c1 - c0 + 1):
            count = (
                scratch[row + 21, column + 21]
                - scratch[row, column + 21]
                - scratch[row + 21, column]
                + scratch[row, column]
            )
            if count >= 10:
                return True
    return False


@njit(cache=True, nogil=True)
def _scan_halo_tiles(ptr, col, nr, nc, size, halo, upper):
    """Find passing core tiles directly from one prepared CSR index stream."""
    nrb = (nr + size - 1) // size
    ncb = (nc + size - 1) // size
    counts = np.zeros(ncb, np.int64)
    touched = np.empty(ncb, np.int64)
    accepted = np.zeros(ncb, np.bool_)
    scratch = np.empty((41, 41), np.int32)
    found = []

    for rb in range(nrb):
        r0 = rb * size
        r1 = min(r0 + size, nr)
        wr0 = max(0, r0 - halo)
        wr1 = min(nr, r1 + halo)
        row_boundary = r0 < halo or r1 + halo > nr
        ntouched = 0

        # Count entries in every halo-expanded column tile for this row tile.
        # With halo < size, one column belongs to at most two such tiles.
        for index in range(ptr[wr0], ptr[wr1]):
            column = col[index]
            base = column // size
            first_cb = max(0, base - 1)
            last_cb = min(ncb - 1, base + 1)
            for cb in range(first_cb, last_cb + 1):
                if upper and cb < rb:
                    continue
                c0 = cb * size
                c1 = min(c0 + size, nc)
                wc0 = max(0, c0 - halo)
                wc1 = min(nc, c1 + halo)
                if wc0 <= column < wc1:
                    if counts[cb] == 0:
                        touched[ntouched] = cb
                        ntouched += 1
                    counts[cb] += 1

        remaining = 0
        for index in range(ntouched):
            cb = touched[index]
            c0 = cb * size
            c1 = min(c0 + size, nc)
            col_boundary = c0 < halo or c1 + halo > nc
            if row_boundary or col_boundary:
                # Preserve the old rule: retain non-empty chromosome-boundary tiles.
                accepted[cb] = True
            elif counts[cb] >= 10:
                remaining += 1

        # Find 2x2 squares and offer each one to all expanded tiles containing it.
        for row in range(wr0, wr1 - 1):
            if remaining == 0:
                break
            x = ptr[row]
            x_end = ptr[row + 1]
            y = x_end
            y_end = ptr[row + 2]
            if x_end - x < 2 or y_end - y < 2:
                continue

            previous = -2
            while x < x_end and y < y_end:
                left = col[x]
                right = col[y]
                if left < right:
                    x += 1
                elif right < left:
                    y += 1
                else:
                    if left == previous + 1:
                        first_cb = max(0, previous // size - 1)
                        last_cb = min(ncb - 1, left // size + 1)
                        for cb in range(first_cb, last_cb + 1):
                            if upper and cb < rb:
                                continue
                            if counts[cb] < 10 or accepted[cb]:
                                continue
                            c0 = cb * size
                            c1 = min(c0 + size, nc)
                            wc0 = max(0, c0 - halo)
                            wc1 = min(nc, c1 + halo)
                            if wc0 <= previous and left < wc1:
                                if _halo_local_pass(
                                    ptr, col, row, previous,
                                    wr0, wr1, wc0, wc1, scratch,
                                ):
                                    accepted[cb] = True
                                    remaining -= 1
                    previous = left
                    x += 1
                    y += 1
                    if remaining == 0:
                        break

        for index in range(ntouched):
            cb = touched[index]
            if accepted[cb]:
                found.append((r0, r1, cb * size, min((cb + 1) * size, nc)))
            counts[cb] = 0
            accepted[cb] = False

    output = np.empty((len(found), 4), np.int64)
    for index in range(len(found)):
        for coordinate in range(4):
            output[index, coordinate] = found[index][coordinate]
    return output


def prepare_sparse_support(matrix):
    """Return canonical finite/non-zero occupancy for one complete CSR matrix."""
    if not sp.issparse(matrix) or matrix.format != "csr":
        raise TypeError("Expected a scipy CSR matrix or CSR array")
    support = matrix.copy()
    support.sum_duplicates()
    support.sort_indices()
    occupied = np.isfinite(support.data) & (support.data != 0)
    support.data = occupied.astype(np.uint8, copy=False)
    support.eliminate_zeros()
    return support


def find_valid_halo_tiles(matrix, tile_size=2048, *, upper_triangle=False):
    """Return core bounds whose halo contains a qualifying centered 21x21.

    Interior tiles pass only when a 21x21 window centered inside the core has
    at least 10 occupied positions and a fully occupied 2x2. Non-empty outer
    boundary tiles are retained, matching the previous per-tile behavior.
    """
    if not sp.issparse(matrix) or matrix.format != "csr":
        raise TypeError("Expected a scipy CSR matrix or CSR array")
    size = operator.index(tile_size)
    if size < 21:
        raise ValueError("tile_size must be at least 21")
    halo = 10
    if halo >= size:
        raise ValueError("tile_size must be larger than the 10-bin halo")
    nr, nc = matrix.shape
    if upper_triangle and nr != nc:
        raise ValueError("upper_triangle=True requires a square matrix")
    if nr == 0 or nc == 0 or matrix.nnz == 0:
        return np.empty((0, 4), np.int64)

    output = _scan_halo_tiles(
        matrix.indptr, matrix.indices, nr, nc, size, halo, upper_triangle,
    )
    if len(output) > 1:
        order = np.lexsort((output[:, 2], output[:, 0]))
        output = output[order]
    return output


def iter_grid_tiles(n_rows, n_cols, tile_size, upper_triangular_only):
    """Yield the original non-overlapping core tile grid."""
    for r0 in range(0, n_rows, tile_size):
        r1 = min(r0 + tile_size, n_rows)
        c_start = r0 if upper_triangular_only else 0
        for c0 in range(c_start, n_cols, tile_size):
            c1 = min(c0 + tile_size, n_cols)
            if upper_triangular_only and c1 <= r0:
                continue
            yield r0, r1, c0, c1


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
    D = np.minimum(D, exp.size - 1)
    
    denom = exp[D]  # (h,w)
    out = np.divide(block_dense, denom, out=block_dense.copy(),
                    where=(denom != 0)).astype(np.float32, copy=False)

    return out


def block_normalize(block_dense, exp, R0, C0, k=21):
    """
    Apply optional cis O/E normalization followed by local min-max normalization.
    """
    if exp is not None:
        exp = np.asarray(exp, dtype=np.float32)
        block_dense = distance_normalize_block(block_dense, exp, R0, C0)

    block_tf = tf.convert_to_tensor(block_dense, dtype=tf.float32)
    return local_minmax_normalize_2d(block_tf, k=k).numpy()


def iter_csr_tiles(M, tile_size=2048, k=21, exp=None,
                   upper_triangular_only=False, sparse_prefilter=True):
    """Traverse halo-extended CSR tiles and normalize only retained tiles.

    With sparse_prefilter=True and k=21, the complete CSR index stream is
    scanned once to identify passing core bounds. CSR submatrices are then
    constructed only for those retained bounds.
    """
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()

    n_rows, n_cols = M.shape
    halo = (k - 1) // 2
    if exp is not None:
        exp = np.asarray(exp, dtype=np.float32)

    if sparse_prefilter and k == 21:
        support = prepare_sparse_support(M)
        tile_bounds = find_valid_halo_tiles(
            support,
            tile_size=tile_size,
            upper_triangle=upper_triangular_only,
        )
        del support
    else:
        tile_bounds = iter_grid_tiles(
            n_rows, n_cols, tile_size, upper_triangular_only,
        )

    for r0, r1, c0, c1 in tile_bounds:
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        R0 = max(0, r0 - halo)
        R1 = min(n_rows, r1 + halo)
        C0 = max(0, c0 - halo)
        C1 = min(n_cols, c1 + halo)

        pad_top = max(0, halo - r0)
        pad_bottom = max(0, r1 + halo - n_rows)
        pad_left = max(0, halo - c0)
        pad_right = max(0, c1 + halo - n_cols)

        block = M[R0:R1, C0:C1]
        if block.nnz == 0:
            continue

        block_dense = block.toarray().astype(np.float32, copy=False)
        block_dense = np.nan_to_num(
            block_dense, nan=0.0, posinf=0.0, neginf=0.0,
        )
        if block_dense.sum() == 0:
            continue

        block_norm = block_normalize(block_dense, exp, R0, C0, k=k)

        if any((pad_top, pad_bottom, pad_left, pad_right)):
            block_norm = np.pad(
                block_norm,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        rr0 = (r0 - R0) + pad_top
        rr1 = rr0 + (r1 - r0)
        cc0 = (c0 - C0) + pad_left
        cc1 = cc0 + (c1 - c0)

        yield (
            r0, r1, c0, c1,
            block_norm,
            rr0, rr1, cc0, cc1,
        )

def iter_cooler_scan_candidates(cool_path, resolutions, chroms, expected_values,
                                balance, base_fcn, tile_size=2048, k=21, cutoff=0.3,
                                sparse_prefilter=True):
    
    candidates = {}
    count = 0
    for res in resolutions:
        clr = cooler.Cooler('{0}::resolutions/{1}'.format(cool_path, res))
        candidates[res] = defaultdict(list)
        # cis
        for chrom in chroms:
            log.info('  Scanning {0} at resolution {1} ...'.format(chrom, res))
            M = clr.matrix(balance=balance, sparse=True).fetch(chrom).tocsr()
            for r0, r1, c0, c1, block_norm, rr0, rr1, cc0, cc1 in iter_csr_tiles(
                M,
                tile_size=tile_size,
                k=k,
                exp=expected_values[res][chrom],
                upper_triangular_only=True,
                sparse_prefilter=sparse_prefilter
            ):
                x2d = tf.convert_to_tensor(block_norm, dtype=tf.float32)
                p_sv_full = fcn_sv_probability_map(base_fcn, x2d).numpy()
                p_sv_np = p_sv_full[rr0:rr1, cc0:cc1]
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
                for r0, r1, c0, c1, block_norm, rr0, rr1, cc0, cc1 in iter_csr_tiles(
                    M,
                    tile_size=tile_size,
                    k=k,
                    exp=None,
                    upper_triangular_only=False,
                    sparse_prefilter=sparse_prefilter
                ):
                    x2d = tf.convert_to_tensor(block_norm, dtype=tf.float32)
                    p_sv_full = fcn_sv_probability_map(base_fcn, x2d).numpy()
                    p_sv_np = p_sv_full[rr0:rr1, cc0:cc1]
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
            if chrom1 == chrom2:
                exp = expected_values[res][chrom1]
            else:
                exp = None

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
