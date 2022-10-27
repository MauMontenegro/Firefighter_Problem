import numpy as np
import matplotlib.pyplot as plt

dp_results=np.load('Experiments/Experiment_0_20s/Statistics_DP.npy')
#iqp_results=np.load('Experiments/Statistics_IQP.npy')
#ilp_results=np.load('Experiments/Statistics_ILP.npy')

#np.save(path / "Statistics_DP", np.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
fig, ax = plt.subplots(1)
y= np.arange(0, len(dp_results[2]), 1, dtype=int)
ax.plot(y, dp_results[2], label="Mean Time Vertices DP", color="red")
#ax.fill_between(y, time_mean + time_std_dv/2, time_mean - time_std_dv/2, facecolor="red", alpha=0.5)
plt.savefig("Experiments/Experiment_0_20s/Statistics_DP.npy/DP_Time_test.png")