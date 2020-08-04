using DelimitedFiles
using Statistics
using DecisionTree

datafile="D:/Explore/DataSets/iris/iris.data"
iris=readdlm(datafile,',')
label = string.(iris[:,5])
input = float.(iris[:, 1:4])

model = DecisionTreeClassifier(max_depth=2)
fit!(model, input[1:2:end,:], label[1:2:end])
# 显示模型中的类别
println(get_classes(model))

# 预测偶数样本的类别
class=predict(model, input[2:2:end,:])
print("正确率=",mean(class.==label[2:2:end]))

# 给出输入样本各个类别的概率
predict_proba(model, [5.9,3.0,5.1,1.9])

model = build_tree(label,input) #创建决策树
model = prune_tree(model, 0.9)  #剪枝
print_tree(model, 5) #打印模型
apply_tree(model, [5.9,3.0,5.1,1.9]) #预测
apply_tree_proba(model, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])


n_folds=3
accuracy = nfoldCV_tree(label, input, n_folds)

# 随机森林分类
model = build_forest(label, input, 2, 10, 0.5, 6)
apply_forest(model, [5.9,3.0,5.1,1.9])
apply_forest_proba(model, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(label, input, n_folds, n_subfeatures)