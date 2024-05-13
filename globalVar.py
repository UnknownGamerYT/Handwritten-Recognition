# This code was accessed from the following GitHub: https://github.com/lcukerd/Suleman/tree/master
# We have used the code produced by Suleman et al. for extracting text lines from text

import numpy as np

def init(lines, centroids):
    global dists
    dists = np.zeros((len(lines), len(centroids))) -1
    dists.shape

def check(msg):
    global checkmsg
    checkmsg = msg