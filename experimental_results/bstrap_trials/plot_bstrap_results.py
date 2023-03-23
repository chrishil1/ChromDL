import matplotlib.pyplot as plt
import numpy as np

# auROC subplot

aurocs = [0.93258267, 0.938376441, 0.94173732, 0.947584925]
auroc_pos = [0.5, 1, 1.5, 0]

ax1 = plt.subplot()
ax1.scatter(auroc_pos[0], aurocs[0], c='b')
ax1.scatter(auroc_pos[1], aurocs[1], c='r')
ax1.scatter(auroc_pos[2], aurocs[2], c='orange')
ax1.scatter(auroc_pos[3], aurocs[3], c='g')
ax1.set_ylabel("Bootstrapped Mean auROC")
ax1.set_ylim(0.925, 0.955)

pos = [0.5, 1, 1.5, 0]
bounds = [(0.929806437, 0.935309366), (0.935756201, 0.940953627), (0.93919717, 0.944223183), (0.945035217, 0.95007591)]

for z in range(len(pos)):
    x = pos[z]
    y = (bounds[z][0] + bounds[z][1]) / 2
    yerr = (bounds[z][1]) - y
    
    if z == 0:
        c = 'b'
    elif z == 1:
        c = 'r'
    elif z == 2:
        c = 'orange'
    else:
        c = 'g'
    ax1.errorbar(x=x, y=y, yerr=yerr, capsize=3, c=c)

# auPRC subplot

auprcs = [0.34252532, 0.37090131, 0.37936217, 0.40137059]
auprc_pos = [3, 3.5, 4, 2.5]

ax2 = ax1.twinx()
ax2.scatter(auprc_pos[3], auprcs[3], c='g', label='ChromDL')
ax2.scatter(auprc_pos[0], auprcs[0], c='b', label='DeepSEA')
ax2.scatter(auprc_pos[1], auprcs[1], c='r', label='DanQ')
ax2.scatter(auprc_pos[2], auprcs[2], c='orange', label='DanQ-JASPAR')
ax2.set_ylim(0.31, 0.43)
ax2.set_yticks([0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43])
ax2.set_ylabel("Bootstrapped Mean auPRC")

pos = [3, 3.5, 4, 2.5]
bounds = [(0.3322318,0.35282234), (0.36045311,0.38128215), (0.36893156,0.38979663), (0.39114316,0.41161951)]

for z in range(len(pos)):
    x = pos[z]
    y = (bounds[z][0] + bounds[z][1]) / 2
    yerr = (bounds[z][1]) - y
    
    if z == 0:
        c = 'b'
    elif z == 1:
        c = 'r'
    elif z == 2:
        c = 'orange'
    else:
        c = 'g'
    ax2.errorbar(x=x, y=y, yerr=yerr, capsize=3, c=c)

plt.grid(axis="y", which="both", alpha=0.3)

x = np.array([0.75,3.25])
my_xticks = ['auROC','auPRC']
plt.xticks(x, my_xticks)

plt.legend(fontsize=9, loc='lower center')
plt.savefig("bstrap_results.png")