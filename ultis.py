import numpy as np

def gaussianKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float_)
    centerX, centerY = M/2, N/2

    for u in range(M):
        for v in range(N):
            dist = (u - centerX)**2 + (v - centerY)**2
            H[u, v] = np.exp(-dist/(2*D0**2))

    return H