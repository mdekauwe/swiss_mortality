
import numpy as np

root_beta = 0.93#0.943 # 0.99

zse = np.array([.022, .058, .134, .189, 1.325, 2.872])
froot = np.zeros(6)

# calculate vegin%froot from using rootbeta and soil depth
# (Jackson et al. 1996, Oceologica, 108:389-411)
total_depth = 0.0
for i in range(5):
    total_depth += zse[i] * 100.0  # unit in centimetres
    froot[i] = min(1.0, 1.0 - root_beta**total_depth)
froot[5] = 1.0 - froot[4]

for i in range(4, 0, -1):
    froot[i] = froot[i] - froot[i-1]


for i in range(6):
    print(i, froot[i])

# empty two bottom layers

#froot[3] += froot[4] + froot[5]
#froot[4] = 0.0
#froot[5] = 0.0

print("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f" % (froot[0], froot[1], froot[2], froot[3], froot[4], froot[5]))
print(np.sum(froot))
