#!/usr/bin/env python3
"""
Calls the QR-PerfTest executable with many different matrix sizes and plots the
comparison between the naive implementation and Eigen's implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from subprocess import check_output
from os.path import dirname, realpath, join
import re

dir_path = dirname(realpath(__file__))


def test(n):
    N = max(1, int(round(100000 / n)))
    results = np.zeros((4, N))
    for i in range(N):
        executable = join(dir_path, '..', '..', 'build', 'examples',
                          'QR-PerfTest', 'QR-PerfTest')
        result = check_output([executable, str(n)]).decode('utf-8')
        float_pattern = re.compile(r'\d+(.\d+)?(?:e[+-]?\d+)?')
        results[:, i] = np.array([
            float(re.search(float_pattern, line).group(0))
            for line in result.splitlines()
        ])
    return np.mean(results, axis=-1)


N = 50
custom_times = np.zeros((N, 1))
eigen_times = np.zeros((N, 1))
ns = np.logspace(1, np.log10(120), N)
for i in range(N):
    result = test(ns[i])
    custom_times[i] = result[0]
    eigen_times[i] = result[2]

plt.figure(figsize=(6, 5))
plt.plot(ns, custom_times, label='Naive')
plt.plot(ns, eigen_times, label='Eigen')
plt.legend()
plt.xlabel('Matrix size')
plt.ylabel('Execution time [s]')
plt.savefig('linear.svg')

plt.figure(figsize=(6, 5))
plt.loglog(ns, custom_times, label='Naive')
plt.loglog(ns, eigen_times, label='Eigen')
plt.legend()
plt.xlabel('Matrix size')
plt.ylabel('Execution time [s]')
plt.savefig('loglog.svg')

plt.show()
