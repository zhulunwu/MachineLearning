using LIBSVM, Test
using DelimitedFiles
using SparseArrays

#Regression tests
whiteside, hdr = readdlm(joinpath("D:/Explore/DataSets/whiteside/whiteside.csv"), ',', header=true)
ws = convert(Matrix{Float64}, whiteside[:,2:3])
X = Array{Float64, 2}(ws[:, 2]')
y = ws[:, 1]

m = svmtrain(X, y, svmtype = EpsilonSVR, cost = 10., gamma = 1.)
yeps, d = svmpredict(m, X)
@test sum(yeps - y) ≈ 7.455509045783046


nu1 = svmtrain(X, y, svmtype = NuSVR, cost = 10.0, nu = 0.7, gamma = 2.0, tolerance = 0.001)
ynu1, d = svmpredict(nu1, X)
@test sum(ynu1 - y) ≈  14.184665717092

nu2 = svmtrain(X, y, svmtype = NuSVR, cost = 10.0, nu = 0.9)
ynu2, d =svmpredict(nu2, X)
@test sum(ynu2 - y) ≈ 6.686819661799177

# Multithreading testing

# Assign by maximum number of threads
ntnu1 = svmtrain(X, y, svmtype = NuSVR, cost = 10.0,
                nu = 0.7, gamma = 2.0, tolerance = 0.001,
                nt = -1)
ntynu1, ntd = svmpredict(ntnu1, X)
@test sum(ntynu1 - y) ≈  14.184665717092

# Assign by environment
ENV["OMP_NUM_THREADS"] = 2

ntm = svmtrain(X, y, svmtype = EpsilonSVR, cost = 10.0, gamma = 1.0)
ntyeps, ntd = svmpredict(m, X)
@test sum(ntyeps - y) ≈ 7.455509045783046
