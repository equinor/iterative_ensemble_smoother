import numpy as np
import matplotlib.pyplot as plt

nparam = 1000
xinc = 1.0
nreal = 100
field_relative_range = 0.05
obs_std = 1.0
#nparam = 2000
local = True
dist = False
global_update = False
Nlist = [0,1]
#Nlist = [0,1,2]
N = 2
nobs_in_file = 200
use_simple_kriging = False

case = [
    "N_100_std_1.83_obsrange_0.00_nobs_200_exponential",
    "N_100_std_1.0_obsrange_0.03_nobs_200_exponential",
]
# case = [
#     "N_100_std_2.58_obsrange_0.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.01_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.02_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.03_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.05_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.10_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.20_nobs_200_exponential",
# ]
obs_range = [
    0.00,
    0.03,
]
# obs_range = [
#     0.00,
#     0.01,
#     0.02,
#     0.03,
#     0.05,
#     0.10,
#     0.20,
#     0.30,
#     0.40,
#     0.60,
#     0.90,

# ]
# case = [
#     "N_100_std_2.58_obsrange_0.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.10_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.20_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.30_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.40_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.50_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.60_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.70_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.80_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_0.90_nobs_200_exponential",    
# ]

# case = [
#     "N_100_std_2.58_obsrange_0.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_1.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_2.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_3.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_4.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_5.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_6.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_7.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_8.00_nobs_200_exponential",
#     "N_100_std_2.58_obsrange_9.00_nobs_200_exponential",
# ]

refcase = [
    "std_2.58_obsrange_0.0001_nobs_25",
]

x = np.arange(nparam) * xinc
y_local = []
y_local_std = []
y_dist = []
y_dist_std = []
y_global = []
y_global_std = []
y_prior = []
y_diff_dist = []
y_diff_local = []
y_diff_global =[]
y_diff_sk =[]
y_diff_sk_global =[]
w = []
for i in range(N):
#    nreal = realizations[i]
    case_name = case[i]
    if global_update:
        filename = "X_post_mean_global_" + case_name + ".csv"
        y_global.append(np.loadtxt(filename,delimiter=","))
        filename = "X_post_std_global_" + case_name + ".csv"
        y_global_std.append(np.loadtxt(filename,delimiter=","))

    if dist:
        filename = "X_post_mean_dist_" + case_name + ".csv"
        y_dist.append(np.loadtxt(filename,delimiter=","))
        filename = "X_post_std_dist_" + case_name + ".csv"
        y_dist_std.append(np.loadtxt(filename,delimiter=","))
    if local:
        filename = "X_post_mean_local_" + case_name + ".csv"
        y_local.append(np.loadtxt(filename,delimiter=","))
        filename = "X_post_std_local_" + case_name + ".csv"
        y_local_std.append(np.loadtxt(filename,delimiter=","))

    

    filename = "X_prior_mean_" + case_name + ".csv"
    y_prior.append(np.loadtxt(filename,delimiter=","))

#    filename = "X_diff_dist_prior_post_mean_" + case_name + ".csv"
#    y_diff_dist.append(np.loadtxt(filename,delimiter=","))
#    filename = "X_diff_local_prior_post_mean_" + case_name + ".csv"
#    y_diff_local.append(np.loadtxt(filename,delimiter=","))
#    filename = "X_diff_mean_global_" + case_name + ".csv"
#    y_diff_global.append(np.loadtxt(filename,delimiter=","))

#    filename = "X_diff_mean_sk_" + case_name + ".csv"
#    y_diff_sk.append(np.loadtxt(filename,delimiter=","))
#    filename = "X_diff_mean_global_sk_" + case_name + ".csv"
#    y_diff_sk_global.append(np.loadtxt(filename,delimiter=","))
    if use_simple_kriging:
        filename = "X_post_SK_" + case_name + ".csv"
        w.append(np.loadtxt(filename,delimiter=","))


#obs_vector = np.array([5.0, -5.0, 4.0, 5.0], dtype=np.float64)
#obs = 50
obs_filename = "obs_" + str(nobs_in_file) +  ".csv"
print(f"Read file: {obs_filename}")
obs_vector = np.loadtxt(obs_filename, delimiter=",")

obs_index_filename = "obs_index_" + str(nobs_in_file) +  ".csv"
print(f"Read file: {obs_index_filename}")
obs_index_vector = np.loadtxt(obs_index_filename, delimiter=",")

xpositions = (obs_index_vector) * xinc
nobs = obs_vector.shape[0]
print(f"nobs: {nobs}")


fig,ax = plt.subplots()
for i in Nlist:
    if local:
#        ax.plot(x,y_prior[i],label="local_prior_" + case[i])
        #ax.plot(x,y_local[i],label="local_" + case[i])
        ax.plot(x,y_local[i],label=f"Relative correlation range for observations: {obs_range[i]}")
    if dist:
        ax.plot(x,y_dist[i],label="localdist_" + case[i])
    if global_update:
        ax.plot(x,y_global[i],label="global_" + case[i])
#    ax.plot(x,w[i],label="Reference " + refcase[i])
ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
plt.title(f"Mean posterior  Nreal={nreal}  Field relative correlation range: {field_relative_range}  Nobs: {nobs} Obs std: {obs_std}")
#plt.title("Global and Local mean posterior")
ax.legend()
plt.show()

fig,ax = plt.subplots()
for i in Nlist:
    if local:
        #ax.plot(x,y_local_std[i],label="local_" + case[i])
        ax.plot(x,y_local_std[i],label=f"Relative correlation range for observations: {obs_range[i]}")
        ax.set_ylim(0.0, 1.5)
    if dist:
        ax.plot(x,y_dist_std[i],label="localdist_" + case[i])
    if global_update:
        ax.plot(x,y_global_std[i],label="global_" + case[i])

#plt.title("Global and Local std posterior")
plt.title(f"Stdev posterior  Nreal={nreal}  Field relative correlation range: {field_relative_range}  Nobs: {nobs} Obs std: {obs_std}")
ax.legend()
plt.show()

fig,ax = plt.subplots()
for i in Nlist:
    ax.plot(x,y_global[i],label=case[i])
    ax.plot(x,y_global[i] + y_global_std[i],linestyle='solid', color="black")
    ax.plot(x,y_global[i] - y_global_std[i],linestyle='solid', color="black")
    ax.plot(x,w[i],label="Reference " + refcase[i])
ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
plt.title("Global posterior")
ax.legend()
plt.show()

fig,ax = plt.subplots()
for i in Nlist:

    ax.plot(x,y_local[i],label=case[i])
    ax.plot(x,y_local[i] + y_local_std[i],linestyle='solid', color="black")
    ax.plot(x,y_local[i] - y_local_std[i],linestyle='solid', color="black")
    ax.plot(x,w[i],label="Reference " + refcase[i])
ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
plt.title("Local posterior")
ax.legend()
plt.show()

# fig,ax = plt.subplots()
# ax.plot(x,y_diff_local[0],label="Difference post-prior DL nreal= 100")
# ax.plot(x,y_diff_local[1],label="Difference post-prior DL nreal= 500")
# ax.plot(x,y_diff_local[2],label="Difference post-prior DL nreal= 40000")
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(x,y_diff_global[0],label="Difference post-prior Global nreal= 100")
# ax.plot(x,y_diff_global[1],label="Difference post-prior Global nreal= 500")
# ax.plot(x,y_diff_global[2],label="Difference post-prior Global nreal= 40000")
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(x,y_diff_sk[0],label="Difference post-DL and SK nreal= 100")
# ax.plot(x,y_diff_sk[1],label="Difference post-DL and SK nreal= 500")
# ax.plot(x,y_diff_sk[2],label="Difference post-DL and SK nreal= 40000")
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(x,y_diff_sk_global[0],label="Difference post-Global and SK nreal= 100")
# ax.plot(x,y_diff_sk_global[1],label="Difference post-Global and SK nreal= 500")
# ax.plot(x,y_diff_sk_global[2],label="Difference post-Global and SK nreal= 40000")
# ax.legend()
# plt.show()
