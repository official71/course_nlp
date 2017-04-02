import json
from pprint import pprint
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict


######################
# pos_visualization  #
######################
def pos_visualization(jdata, plt_indx):
    pos_names = []
    pos_list = []
    for p in jdata['pos'].keys():
        pos_names.append(p)
        pos_list.append(jdata['pos'][p])
    pos_x = np.array(pos_list)
    pca = PCA(n_components = 2)
    pos_x_pca = pca.fit_transform(pos_x)
    # pprint(pos_x_pca)

    # fig, ax = plt.subplots()
    plt.figure(plt_indx)
    plt.scatter(pos_x_pca[:,0], pos_x_pca[:,1], alpha=0.5)

    for name, x, y in zip(pos_names, pos_x_pca[:,0], pos_x_pca[:,1]):
        plt.annotate(name, xy=(x, y))

    plt.title('PCA of POS tags')
    # fig.tight_layout()

    plt.savefig("pos_visualization1.png")

######################
# verb_visualization #
######################
def verbs(fn):
    dv = defaultdict(int)
    with open(fn) as fh:
        for line in fh:
            tok = line.strip().split()
            if not tok:
                continue
            if 'VB' in tok[4]:
                dv[tok[1]] = 1
    return dv

def verb_visualization(jdata, plt_indx):
    # all words
    words = []
    word_list = []
    for w in jdata['word'].keys():
        words.append(w)
        word_list.append(jdata['word'][w])
    word_x = np.array(word_list)
    pca = PCA(n_components = 2)
    word_x_pca = pca.fit_transform(word_x)

    # verbs
    dv = verbs('data/english/train.conll')

    plt.figure(plt_indx)
    # fig, ax = plt.subplots()
    for word, x, y in zip(words, word_x_pca[:,0], word_x_pca[:,1]):
        if dv[word]:
            plt.scatter(x, y, alpha=0.5)
            plt.annotate(word, xy=(x, y))
    plt.title('PCA of words (verbs only)')
    # fig.tight_layout()

    plt.savefig("verb_visualization1.png")

if __name__ == '__main__':
    with open('output/model_pos_embeddings.json') as jfile:
        jdata = json.load(jfile)

    plt_indx = 0
    pos_visualization(jdata, plt_indx)

    plt_indx += 1
    verb_visualization(jdata, plt_indx)

    plt.show()
