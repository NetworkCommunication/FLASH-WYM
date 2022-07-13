import numpy as np

def get_mask(data, method="norm"):
    mask = np.ones(data.shape)
    max = data.sum(axis=1).max() / 3
    for i in range(data.shape[0]):
        choice = np.array([0, 1])
        length = data[i].sum()
        prob_1 = 0.2
        if method == "norm":
            prob_1 = 0.5
        if method == "user":
            prob_1 += ((0.8 / max) * length)
        if method == "user_inverse":
            prob_1 = 0.75
            prob_1 -= ((0.6 / (max)) * length)
        if method == "none":
            prob_1 = 1
        if method == "all":
            prob_1 = 0
        if prob_1 > 0.8:
            prob_1 = 0.8
        if prob_1 < 0.2:
            prob_1 = 0.2
        prob_0 = 1 - prob_1
        proba = np.random.choice(choice, data.shape[1] - length, p=[prob_0, prob_1])
        mask[i][data[i]==0] = proba
    return mask

