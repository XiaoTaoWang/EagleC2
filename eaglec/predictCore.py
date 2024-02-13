import joblib, glob, os, cooler
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.cluster import dbscan
from eaglec.extractMatrix import check_sparsity
from eaglec.utilities import distance_normaize_core, image_normalize

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

def predict(cache_folder, models, prob_cutoff=0.75, batch_size=256):

    queue = glob.glob(os.path.join(cache_folder, 'collect_*.pkl'))
    original_predictions = {}
    SV_labels = ['++', '+-', '-+', '--', '++/--', '+-/-+']
    for q in queue:
        data = joblib.load(q)
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
    
    return original_predictions

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
                    mappable[(tr, qr)][sv] = defaultdict(set)
                    mappable[(qr, tr)][sv] = defaultdict(set)
                    for k in by_res[tr][sv]:
                        if k in by_res[qr][sv]:
                            for tx, ty in by_res[tr][sv][k]:
                                s_l = range(tx*tr//qr, int(np.ceil((tx+1)*tr/qr)))
                                e_l = range(ty*tr//qr, int(np.ceil((ty+1)*tr/qr)))
                                for x in s_l:
                                    for y in e_l:
                                        if (x, y) in by_res[qr][sv][k]:
                                            mappable[(tr, qr)][sv][k].add((tx, ty))
                                            mappable[(qr, tr)][sv][k].add((x, y))
    
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

    return new

def dict2list(D, res):

    L = []
    for sv in D:
        for c1, c2 in D[sv]:
            for p1, p2 in D[sv][(c1, c2)]:
                line = (c1, p1*res, c2, p2*res) + tuple(D[sv][(c1, c2)][(p1, p2)]) + (res, res)
                L.append(line)
    
    return L

def cluster_SVs(preSVs, r=15000):

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

def refine_predictions(by_res, resolutions, models, mcool, balance, exp,
                       w=15, baseline_prob=0.01):

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
            nL = []
            for line in L:
                c1, p1, c2, p2 = line[:4]
                info = list(line[4:])
                s_l = range((p1-tr)//qr, int(np.ceil((p1+tr*2)/qr)))
                e_l = range((p2-tr)//qr, int(np.ceil((p2+tr*2)/qr)))
                images = []
                coords = []
                for x in s_l:
                    for y in e_l:
                        if c1 == c2:
                            if y - x < 8:
                                continue

                        interval1 = (c1, x*qr-qr*w, x*qr+qr*w+qr)
                        interval2 = (c2, y*qr-qr*w, y*qr+qr*w+qr)
                        M = clr.matrix(balance=balance, sparse=False).fetch(interval1, interval2)
                        M[np.isnan(M)] = 0

                        if M.max() == M.min():
                            continue

                        if c1 == c2:
                            M = distance_normaize_core(M, exp[qr], x, y, w)
                        
                        M = image_normalize(M)
                        images.append(M)
                        coords.append((c1, x*qr, c2, y*qr))
                
                if len(images):
                    images = np.r_[images]
                    images = convert2TF(images, 256)
                    prob_pool = np.stack([model.predict(images) for model in models])
                    idx = np.argmax(info[:-2])
                    prob_mean = prob_pool.mean(axis=0)[:,idx]
                    best_i = prob_mean.argmax()
                    if prob_mean[best_i] > baseline_prob:
                        info[-1] = qr
                        nL.append(coords[best_i] + tuple(info))
                    else:
                        sv_list.append(line)
                else:
                    sv_list.append(line)
            
            tr = qr
            L = nL
        
        sv_list.extend(L)
    
    by_class = defaultdict(dict)
    for line in sv_list:
        c1, p1, c2, p2 = line[:4]
        prob_ = line[4:-1]
        key = (c1, c2)
        by_class[key][(p1, p2)] = {'prob':(max(prob_), sum(prob_)), 'record':line}
    
    SVs = cluster_SVs(by_class, r=1.5*res_ref[-1])

    return SVs
    