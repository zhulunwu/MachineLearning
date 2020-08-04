#使用svm进行分类
using DelimitedFiles
using LIBSVM
using Statistics
using SparseArrays

datafile="D:/Explore/DataSets/iris/iris.data"
iris=readdlm(datafile,',')
label = iris[:,5]
input = convert(Matrix{Float64}, iris[:, 1:4]')
model = svmtrain(input[:, 1:2:end], label[1:2:end]) #奇数样本为训练样本
(class,value) = svmpredict(model, input[:, 2:2:end])

#计算分类结果正确率
print("正确率=",mean(class.==label[2:2:end]))

# 使用稀疏矩阵
model = svmtrain(sparse(input[:, 1:2:end]), label[1:2:end]; verbose=true)
(class,value) = svmpredict(model, input[:, 2:2:end])
print("正确率=",mean(class.==label[2:2:end]))