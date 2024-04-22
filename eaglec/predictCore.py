import joblib, glob, os, cooler
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.cluster import dbscan
from eaglec.extractMatrix import check_sparsity
from eaglec.utilities import distance_normaize_core, image_normalize, get_queue, dict2list, list2dict

def load_models(root_folder):

    model_paths = glob.glob(os.path.join(root_folder, '*'))
    models = []
    for f in model_paths:
        models.append(tf.keras.models.load_model(f))
    
    return models

def transform_dataset(image):

    return (image[:,5:-5,5:-5], image), None

def convert2TF(images, batch_size=256):

    images = images.astype(np.float32)
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.batch(batch_size)
    images = images.map(transform_dataset)
    images = images.cache()

    return images

def predict(cache_folder, models, ref_gaps, prob_cutoff=0.75, max_gap=1, batch_size=256):

    queue = get_queue(cache_folder, maxn=100000)
    original_predictions = {}
    SV_labels = ['++', '+-', '-+', '--', '++/--', '+-/-+']
    for data in queue:
        images = np.r_[[d[0] for d in data]]
        images = convert2TF(images, batch_size)
        coords = [d[1] for d in data]
        prob_pool = np.stack([model.predict(images) for model in models])
        prob_mean = prob_pool.mean(axis=0)[:,:6]
        for i in range(prob_mean.shape[0]):
            c1, p1, c2, p2, res = coords[i]
            prob = prob_mean[i]
            maxi = prob.argmax()
            if prob[maxi] > prob_cutoff:
                sv = SV_labels[maxi]
                if not res in original_predictions:
                    original_predictions[res] = {}
                
                if not sv in original_predictions[res]:
                    original_predictions[res][sv] = {}
                
                if not (c1, c2) in original_predictions[res][sv]:
                    original_predictions[res][sv][(c1, c2)] = {}
                
                original_predictions[res][sv][(c1, c2)][(p1, p2)] = prob
    
    original_predictions = remove_redundant_predictions(original_predictions)
    gap_removed = {}
    for res in original_predictions:
        sv_list = dict2list(original_predictions[res], res)
        clustered = cluster_SVs(sv_list, r=1.5*res)
        gap_removed[res] = list2dict(check_gaps(clustered, ref_gaps, max_gap), res)
    
    return gap_removed

def cross_resolution_mapping(by_res):

    mappable = {}
    resolutions = sorted(by_res, reverse=True)
    for i in range(len(resolutions)-1):
        for j in range(i+1, len(resolutions)):
            tr = resolutions[i]
            qr = resolutions[j]
            mappable[(tr, qr)] = {}
            mappable[(qr, tr)] = {}
            for sv in by_res[tr]:
                if sv in by_res[qr]:
                    mappable[(tr, qr)][sv] = defaultdict(dict)
                    mappable[(qr, tr)][sv] = defaultdict(dict)
                    for k in by_res[tr][sv]:
                        if k in by_res[qr][sv]:
                            for tx, ty in by_res[tr][sv][k]:
                                s_l = range(tx*tr//qr, int(np.ceil((tx+1)*tr/qr)))
                                e_l = range(ty*tr//qr, int(np.ceil((ty+1)*tr/qr)))
                                for x in s_l:
                                    for y in e_l:
                                        if (x, y) in by_res[qr][sv][k]:
                                            # the value records probability at finer resolutions
                                            mappable[(tr, qr)][sv][k][(tx, ty)] = by_res[qr][sv][k][(x, y)]
                                            # the value records probability at coarser resolutions
                                            mappable[(qr, tr)][sv][k][(x, y)] = by_res[tr][sv][k][(tx, ty)]
    
    return mappable

def remove_redundant_predictions(by_res):

    # remove redundant predictions at coarser resolutions
    mapping_table = cross_resolution_mapping(by_res)
    resolutions = sorted(by_res, reverse=True)
    new = {}
    for i in range(len(resolutions)-1):
        tr = resolutions[i]
        for sv in by_res[tr]:
            for k in by_res[tr][sv]:
                for tx, ty in by_res[tr][sv][k]:
                    support = []
                    for j in range(i+1, len(resolutions)):
                        qr = resolutions[j]
                        if not sv in by_res[qr]:
                            support.append(False)
                        else:
                            if not k in by_res[qr][sv]:
                                support.append(False)
                            else:
                                if (tx, ty) in mapping_table[(tr, qr)][sv][k]:
                                    support.append(True)
                                else:
                                    support.append(False)
                    
                    if sum(support) == 0:
                        if not tr in new:
                            new[tr] = {}
                        
                        if not sv in new[tr]:
                            new[tr][sv] = {}
                        
                        if not k in new[tr][sv]:
                            new[tr][sv][k] = {}
                        
                        new[tr][sv][k][(tx, ty)] = by_res[tr][sv][k][(tx, ty)]
    
    new[resolutions[-1]] = by_res[resolutions[-1]]
    '''
    # records the highest probability score across resolutions
    SV_labels = ['++', '+-', '-+', '--', '++/--', '+-/-+']
    for tr in new:
        for sv in new[tr]:
            for k in new[tr][sv]:
                for tx, ty in new[tr][sv][k]:
                    sort_table = [(new[tr][sv][k][(tx, ty)].max(), tuple(new[tr][sv][k][(tx, ty)]))]
                    for pair in mapping_table:
                        if pair[0] != tr:
                            continue
                        if not sv in mapping_table[pair]:
                            continue
                        if not k in mapping_table[pair][sv]:
                            continue
                        if (tx, ty) in mapping_table[pair][sv][k]:
                            sort_table.append((mapping_table[pair][sv][k][(tx, ty)].max(), tuple(mapping_table[pair][sv][k][(tx, ty)])))

                    sort_table.sort(reverse=True)
                    for item in sort_table:
                        prob = np.r_[item[1]]
                        maxi = prob.argmax()
                        if SV_labels[maxi] == sv:
                            new[tr][sv][k][(tx, ty)] = prob
                            break
    '''
    return new

def cluster_SVs(sv_list, r=15000):

    preSVs = defaultdict(dict)
    for line in sv_list:
        c1, p1, c2, p2 = line[:4]
        prob_ = line[4:10]
        key = (c1, c2)
        preSVs[key][(p1, p2)] = {'prob':(max(prob_), sum(prob_)), 'record':line}

    SVs = []
    for chrom in preSVs:
        by_sv = preSVs[chrom]
        sort_list = [(by_sv[key]['prob'], key) for key in by_sv]
        sort_list.sort(reverse=True)
        pos = np.r_[[i[1] for i in sort_list]]
        if len(pos) >= 2:
            pool = set()
            _, labels = dbscan(pos, eps=r, min_samples=2)
            for i, p in enumerate(sort_list):
                if p[1] in pool:
                    continue
                c = labels[i]
                if c==-1:
                    pool.add(p[1])
                    SVs.append(by_sv[p[1]]['record'])
                else:
                    SVs.append(by_sv[p[1]]['record'])
                    sub = pos[labels==c]
                    for q in sub:
                        pool.add(tuple(q))
        else:
            for key in by_sv:
                SVs.append(by_sv[key]['record'])
    
    return SVs

def check_gaps(sv_list, ref_gaps, max_gap=2):

    out = []
    strands = ['++', '+-', '-+', '--', '++/--', '+-/-+']
    for c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2 in sv_list:
        gaps = ref_gaps[res2]
        b1 = p1 // res2
        b2 = p2 // res2
        probs = np.r_[[prob1, prob2, prob3, prob4, prob5, prob6]]
        idx = np.argmax(probs)
        strand = strands[idx]
        gap_x = gap_y = 0
        if strand == '+-':
            gap_x = gaps[c1][(b1+1):(b1+6)].sum()
            gap_y = gaps[c2][(b2-5):b2].sum()
        elif strand == '++':
            gap_x = gaps[c1][(b1+1):(b1+6)].sum()
            gap_y = gaps[c2][(b2+1):(b2+6)].sum()
        elif strand == '-+':
            gap_x = gaps[c1][(b1-5):b1].sum()
            gap_y = gaps[c2][(b2+1):(b2+6)].sum()
        elif strand == '--':
            gap_x = gaps[c1][(b1-5):b1].sum()
            gap_y = gaps[c2][(b2-5):b2].sum()
        
        if gap_x + gap_y <= max_gap:
            out.append([c1, p1, c2, p2, prob1, prob2, prob3, prob4, prob5, prob6, res1, res2, '{0},{1}'.format(gap_x, gap_y)])
    
    return out

def refine_predictions(by_res, resolutions, models, mcool, balance, exp,
                       ref_gaps, max_gap=2, w=15, baseline_prob=0.5):

    res_ref = sorted(resolutions, reverse=True)
    res_queue = sorted(by_res, reverse=True)
    if res_queue[-1] == res_ref[-1]:
        sv_list = dict2list(by_res[res_ref[-1]], res_ref[-1])
    else:
        sv_list = []

    for i in range(len(res_queue)):
        tr = res_queue[i]
        if tr == res_ref[-1]:
            continue

        L = dict2list(by_res[tr], tr)
        for j in range(len(res_ref)):
            qr = res_ref[j]
            if qr >= tr:
                continue
            
            uri = os.path.join('{0}::resolutions/{1}'.format(mcool, qr))
            clr = cooler.Cooler(uri)

            images = []
            coords = []
            index_count = 0
            index_map = defaultdict(list)
            info_map = {}
            line_map = {}
            for k, line in enumerate(L):
                line_map[k] = line
                info_map[k] = list(line[4:])
                index_count = len(images)
                c1, p1, c2, p2 = line[:4]
                s_l = range((p1-tr)//qr, int(np.ceil((p1+tr*2)/qr)))
                e_l = range((p2-tr)//qr, int(np.ceil((p2+tr*2)/qr)))
                for x in s_l:
                    for y in e_l:
                        if c1 == c2:
                            if y - x < 8:
                                continue

                        interval1 = (c1, x*qr-qr*w, x*qr+qr*w+qr)
                        if (interval1[1] < 0) or (interval1[2] >= clr.chromsizes[c1]):
                            continue # boundary check
                        interval2 = (c2, y*qr-qr*w, y*qr+qr*w+qr)
                        if (interval2[1] < 0) or (interval2[2] >= clr.chromsizes[c2]):
                            continue
                        M = clr.matrix(balance=balance, sparse=False).fetch(interval1, interval2)
                        M[np.isnan(M)] = 0

                        if M.max() == M.min():
                            continue

                        if c1 == c2:
                            M = distance_normaize_core(M, exp[qr][c1], x, y, w)
                        
                        M = image_normalize(M)
                        images.append(M)
                        coords.append((c1, x*qr, c2, y*qr))
                        index_map[k].append(index_count)
                        index_count += 1
            
            nL = []
            if len(images):
                images = np.r_[images]
                images = convert2TF(images, 256)
                prob_pool = np.stack([model.predict(images) for model in models])
                prob_mean = prob_pool.mean(axis=0)
                for k in index_map:
                    coords_tmp = [coords[i_] for i_ in index_map[k]]
                    if len(coords_tmp):
                        info = info_map[k]
                        idx = np.argmax(info[:-2])
                        prob_tmp = prob_mean[index_map[k]][:,idx]
                        best_i = prob_tmp.argmax()
                        if prob_tmp[best_i] > baseline_prob:
                            info[-1] = qr
                            nL.append(coords_tmp[best_i] + tuple(info))
                        else:
                            sv_list.append(line_map[k])
                    else:
                        sv_list.append(line_map[k])
            else:
                for k in line_map:
                    sv_list.append(line_map[k])
            
            tr = qr
            L = nL
        
        sv_list.extend(L)
    
    SVs = cluster_SVs(sv_list, r=1.5*res_ref[-1])
    SVs = check_gaps(SVs, ref_gaps, max_gap)

    return SVs
    