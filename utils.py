import json
import numpy as np
import sklearn.metrics


def xpath_get(mydict, path):
    elem = mydict
    try:
        for x in path.strip("/").split("/"):
            try:
                x = int(x)
                elem = elem[x]
            except ValueError:
                elem = elem.get(x)
    except:
        pass
    return elem


def read_jsonl(path, dict_key=None, keep_keys=None):
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            if keep_keys is not None:
                d = {k: xpath_get(d, v) for k, v in keep_keys.items()}
            data.append(d)

    if dict_key is not None:
        data = {xpath_get(x, dict_key): x for x in data}

    return data


def top_k_accuracy(gt, prediction, kvals, sort=True):
    # sort predictions according to similarity
    if sort:
        prediction = sorted(range(len(prediction)), key=lambda k: -prediction[k].item())

    k_accuracy = []
    for k in kvals:
        k_accuracy.append(int(gt in prediction[:k]))

    return k_accuracy


def jaccard_similarity(gt, prediction, weights=None):
    if weights is None:
        weights = np.ones_like(gt)

    correct = np.equal(prediction, gt)
    inter = weights * correct * gt * prediction
    return np.sum(inter) / (np.sum(weights * gt) + np.sum(weights * prediction) - np.sum(inter))


def cosine_similarity(gt, prediction, weights=None):
    if weights is None:
        weights = np.ones_like(gt)

    prediction = np.reshape(prediction, [1, -1])

    if len(gt.shape) < 2:
        gt = np.reshape(gt, [1, -1])

    return np.squeeze(sklearn.metrics.pairwise.cosine_similarity(weights * prediction, weights * gt))
