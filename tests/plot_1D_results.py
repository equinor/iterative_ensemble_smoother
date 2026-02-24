import numpy as np
import matplotlib.pyplot as plt

nparam = 3000
xinc = 1.0
#nparam = 2000
local = True
dist = True
global_update = True
Nlist = [39]
N = 40
nobs_in_file = 500
# realizations = [100, 100, 100, 100, 100, 100, 100]
case = [
#     "Nreal_100_std_0.2_rel_obsrange_0.001",
#    "Nreal_100_std_0.3_rel_obsrange_0.001",
#    "N_40000_std_0.3_rel_obsrange_0.3_no_loc",
#    "N_40000_std_0.3_rel_obsrange_0.001_no_loc",
#    "N_40000_std_0.3_rel_obsrange_0.001",
#    "N_40000_std_0.3_rel_obsrange_0.5",
#    "N_40000_std_0.05_rel_obsrange_0.5",
#    "N_40000_std_0.01_rel_obsrange_0.5",

    "N_40000_std_0.001_rel_obsrange_0.0001",
    "N_40000_std_0.001_rel_obsrange_0.1",
#    "N_40000_std_0.001_rel_obsrange_0.2",
    "N_40000_std_0.001_rel_obsrange_0.3",#
#    "N_40000_std_0.001_rel_obsrange_0.4",
    "N_40000_std_0.001_rel_obsrange_0.45",
#    "N_40000_std_0.001_rel_obsrange_0.49",
    "N_40000_std_0.001_rel_obsrange_0.5_nobs_50",
    "N_40000_std_0.001_rel_obsrange_0.5_nobs_125",
    "N_40000_std_0.001_rel_obsrange_0.5_nobs_200",
    "N_40000_std_0.001_rel_obsrange_0.5_nobs_250",
    "N_40000_std_0.001_rel_obsrange_0.5_nobs_500",
    "N_40000_std_0.001_rel_obsrange_0.3_nobs_200",
    "N_40000_std_0.001_rel_obsrange_0.4_nobs_200",
    "N_40000_std_0.1_rel_obsrange_0.4_nobs_200",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_200",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_200_L_0.1",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_200_noloc",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_500_noloc",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_100_noloc",
    "N_40000_std_0.5_rel_obsrange_0.4_nobs_300_noloc",
    "N_40000_std_0.5_rel_obsrange_0.3_nobs_300_noloc",
    "N_40000_std_0.5_rel_obsrange_0.2_nobs_300_noloc",
    "N_40000_std_0.5_rel_obsrange_0.1_nobs_300_noloc",
    "N_40000_std_0.5_rel_obsrange_0.01_nobs_300_noloc",
    "N_40000_std_0.5_rel_obsrange_0.01_nobs_500_noloc",
    "N_40000_std_0.01_rel_obsrange_0.01_nobs_5_noloc",
    "N_40000_std_0.01_rel_obsrange_0.1_nobs_5_noloc",
    "N_40000_std_0.01_rel_obsrange_0.5_nobs_5_noloc",
    "N_40000_std_0.01_rel_obsrange_1.0_nobs_5_noloc",
    "N_40000_std_0.01_rel_obsrange_1.0_nobs_100_noloc",
    "N_40000_std_0.01_rel_obsrange_1.0_nobs_500_noloc",
    "N_40000_std_0.01_rel_obsrange_0.00001_nobs_500_noloc",
    "N_40000_std_0.01_rel_obsrange_0.00001_nobs_2500_noloc",
    "N_40000_std_0.30_rel_obsrange_0.00001_nobs_2500_noloc",
    "N_40000_std_0.30_rel_obsrange_0.1_nobs_2500_noloc",
    "N_40000_std_0.30_rel_obsrange_0.1_nobs_2500_L_0.1",
    "N_40000_std_0.30_rel_obsrange_0.3_nobs_2500_L_0.1",
    "N_40000_std_0.30_rel_obsrange_0.5_nobs_2500_L_0.1",
    "N_40000_std_0.05_rel_obsrange_0.5_nobs_2500_L_0.1",
    "N_40000_std_0.01_rel_obsrange_0.5_nobs_2500_L_0.1",
    "N_40000_std_0.01_rel_obsrange_0.5_nobs_100_L_0.1",
    "N_40000_std_0.01_rel_obsrange_0.5_nobs_500_L_0.1",
#    "N_40000_std_0.3_rel_obsrange_0.3",

#     "Nreal_100_relstd_0.05",
#     "Nreal_100_relstd_0.10",
#     "Nreal_100_relstd_0.20",
#    "Nreal_100_std_0.1_rel_obsrange_0.3",
#    "Nreal_100_std_0.1_rel_obsrange_0.9",
#     "Nreal_100_relstd_0.40",
#     "Nreal_100_relstd_0.50",
]
#realizations = [40000,40000]
#case = [
#    "Nreal_500_relstd_0.01",
#    "Nreal_500_relstd_0.30",
#]
refcase = [
 #   "Rel.std_0.2_rel_obsrange 0.001",
#    "Rel.std_0.1_rel_obsrange_0.3",
#    "Rel.std_0.3_rel_obsrange_0.3",
#    "Rel.std_0.3_rel_obsrange_0.001",
#    "Rel.std_0.3_rel_obsrange_0.5",
#    "Rel.std_0.05_rel_obsrange_0.5",
#    "Rel.std_0.01_rel_obsrange_0.5",

    "Rel.std_0.001_rel_obsrange_0.0001",
    "Rel.std_0.001_rel_obsrange_0.1",
#    "Rel.std_0.001_rel_obsrange_0.2",
    "Rel.std_0.001_rel_obsrange_0.3",
#    "Rel.std_0.001_rel_obsrange_0.4",
    "Rel.std_0.001_rel_obsrange_0.45",
#    "Rel.std_0.001_rel_obsrange_0.49",
    "Rel.std_0.001_rel_obsrange_0.5_nobs_50",
    "Rel.std_0.001_rel_obsrange_0.5_nobs_125",
    "Rel.std_0.001_rel_obsrange_0.5_nobs_200",
    "Rel.std_0.001_rel_obsrange_0.5_nobs_250",
    "Rel.std_0.001_rel_obsrange_0.5_nobs_500",
    "Rel.std_0.001_rel_obsrange_0.3_nobs_200",
    "Rel.std_0.001_rel_obsrange_0.4_nobs_200",
    "Rel.std_0.1_rel_obsrange_0.4_nobs_200",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_200",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_200_L_0.1",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_200_noloc",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_500_noloc",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_100_noloc",
    "Rel.std_0.5_rel_obsrange_0.4_nobs_300_noloc",
    "Rel.std_0.5_rel_obsrange_0.3_nobs_300_noloc",
    "Rel.std_0.5_rel_obsrange_0.2_nobs_300_noloc",
    "Rel.std_0.5_rel_obsrange_0.1_nobs_300_noloc",
    "Rel.std_0.5_rel_obsrange_0.01_nobs_300_noloc",
    "Rel.std_0.5_rel_obsrange_0.01_nobs_500_noloc",
    "Rel.std_0.01_rel_obsrange_0.01_nobs_5_noloc",
    "Rel.std_0.01_rel_obsrange_0.1_nobs_5_noloc",
    "Rel.std_0.01_rel_obsrange_0.5_nobs_5_noloc",
    "Rel.std_0.01_rel_obsrange_1.0_nobs_5_noloc",
    "Rel.std_0.01_rel_obsrange_1.0_nobs_100_noloc",
    "Rel.std_0.01_rel_obsrange_1.0_nobs_500_noloc",
    "Rel.std_0.01_rel_obsrange_0.00001_nobs_500_noloc",
    "Rel.std_0.01_rel_obsrange_0.00001_nobs_2500_noloc",
    "Rel.std_0.30_rel_obsrange_0.00001_nobs_2500_noloc",
    "Rel.std_0.30_rel_obsrange_0.1_nobs_2500_noloc",
    "Rel.std_0.30_rel_obsrange_0.1_nobs_2500_L_0.1",
    "Rel.std_0.30_rel_obsrange_0.3_nobs_2500_L_0.1",
    "Rel.std_0.30_rel_obsrange_0.5_nobs_2500_L_0.1",
    "Rel.std_0.05_rel_obsrange_0.5_nobs_2500_L_0.1",
    "Rel.std_0.01_rel_obsrange_0.5_nobs_2500_L_0.1",
    "Rel.std_0.01_rel_obsrange_0.5_nobs_100_L_0.1",
    "Rel.std_0.01_rel_obsrange_0.5_nobs_500_L_0.1",
#    "Rel.std_0.3_rel_obsrange_0.3",

]
# refcase = [
#     "Rel.std_0.01",
#     "Rel.std_0.05",
#     "Rel.std_0.10",
#     "Rel.std_0.20",
#     "Rel.std_0.30",
#     "Rel.std_0.40",
#     "Rel.std_0.50",
# ]

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
    filename = "X_post_mean_global_" + case_name + ".csv"
    y_global.append(np.loadtxt(filename,delimiter=","))
    filename = "X_post_std_global_" + case_name + ".csv"
    y_global_std.append(np.loadtxt(filename,delimiter=","))

    filename = "X_post_mean_dist_" + case_name + ".csv"
    y_dist.append(np.loadtxt(filename,delimiter=","))
    filename = "X_post_std_dist_" + case_name + ".csv"
    y_dist_std.append(np.loadtxt(filename,delimiter=","))

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

    filename = "X_post_SK_" + case_name + ".csv"
    w.append(np.loadtxt(filename,delimiter=","))


#obs_vector = np.array([5.0, -5.0, 4.0, 5.0], dtype=np.float64)
#obs = 50
obs_filename = "obs_" + str(nobs_in_file) +  ".csv"
obs_vector = np.loadtxt(obs_filename, delimiter=",")
nobs = obs_vector.shape[0]
print(f"nobs: {nobs}")
#bs_vector = np.zeros(nobs, dtype=np.float64)
xpositions = np.linspace(-1.0,1.0, nobs, endpoint=True)
obs_index_vector = np.linspace(10, nparam -10, nobs, dtype=np.int32)
# fig,ax = plt.subplots()
# ax.plot(x,y_prior[0],label="Prior nreal=100")
# ax.plot(x,y_prior[1],label="Prior nreal=500")
# ax.plot(x,y_prior[2],label="Prior nreal=40000")
# ax.plot(x,w,label="Reference")
# ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
# plt.title("Prior")
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(x,y_global[0],label="Global nreal=100")
# ax.plot(x,y_global[1],label="Global nreal=500")
# ax.plot(x,y_global[2],label="Global nreal=40000")
# ax.plot(x,w,label="Reference")
# ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
# plt.title("Global posterior")
# ax.legend()
# plt.show()

fig,ax = plt.subplots()
for i in Nlist:
    if local:
#        ax.plot(x,y_prior[i],label="local_prior_" + case[i])
        ax.plot(x,y_local[i],label="local_" + case[i])
    if dist:
        ax.plot(x,y_dist[i],label="localdist_" + case[i])
    if global_update:
        ax.plot(x,y_global[i],label="global_" + case[i])
    ax.plot(x,w[i],label="Reference " + refcase[i])
ax.plot(obs_index_vector, obs_vector, 'o',label='Obs')
plt.title("Global and Local mean posterior")
ax.legend()
plt.show()

fig,ax = plt.subplots()
for i in Nlist:
    if local:
        ax.plot(x,y_local_std[i],label="local_" + case[i])
    if dist:
        ax.plot(x,y_dist_std[i],label="localdist_" + case[i])
    if global_update:
        ax.plot(x,y_global_std[i],label="global_" + case[i])

plt.title("Global and Local std posterior")
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
