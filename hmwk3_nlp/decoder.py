import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter
import math

# directions and completeness
L = 0 # left
R = 1 # right
F = 0 # not complete
T = 1 # complete

def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.

    scores - an (n+1) x (n+1) matrix
    gold - the gold arcs
    '''

    #YOUR IMPLEMENTATION GOES HERE
    # raise NotImplementedError
    C = defaultdict(float)
    n = len(scores)
    for m in xrange(1, n):
        for i in xrange(n - m):
            j = i + m
            score_ij = scores[i][j]
            score_ji = scores[j][i]

            # 1
            max_1 = float('-inf')
            max_k_1 = -1
            for k in xrange(i, j):
                v = C[(i, k, R, T)] + C[(k+1, j, L, T)] + score_ji
                if v > max_1:
                    max_1 = v
                    max_k_1 = k
            C[(i, j, L, F)] = max_1

            # 2
            max_2 = float('-inf')
            max_k_2 = -1
            for k in xrange(i, j):
                v = C[(i, k, R, T)] + C[(k+1, j, L, T)] + score_ij
                if v > max_2:
                    max_2 = v
                    max_k_2 = k
            C[(i, j, R, F)] = max_2

            # 3
            max_3 = float('-inf')
            max_k_3 = -1
            for k in xrange(i, j):
                v = C[(i, k, L, T)] + C[(k, j, L, F)]
                if v > max_3:
                    max_3 = v
                    max_k_3 = k
            C[(i, j, L, T)] = max_3

            # 4
            max_4 = float('-inf')
            max_k_4 = -1
            for k in xrange(i+1, j+1):
                v = C[(i, k, R, F)] + C[(k, j, R, T)]
                if v > max_4:
                    max_4 = v
                    max_k_4 = k
            C[(i, j, R, T)] = max_4

    # for i in xrange(n):
    #     for j in xrange(i, n):
    #         print "%d, %d, %f" % (i, j, C[(i, j, R, T)])
    # print scores[0]
    # print scores[n - 1]
    # print C[(0, n-1, R, T)]

    exit(0)
