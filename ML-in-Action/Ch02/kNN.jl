using NearestNeighbors
using DelimitedFiles
using Statistics

function autoNorm(features)
     vmin=minimum(features,dims=1)
     vmax=maximum(features,dims=1)
     vrange=vmax-vmin
     return (features.-vmin)./vrange
end

# 读入数据
data=readdlm("datingTestSet.txt")
# 分离特征数据以及标签数据
features=float.(data[:,1:3])
labels=string.(data[:,4])
# 归一化数据
features_norm=autoNorm(features)'
# 划分树数据以及待分类数据
len=Int(0.5*length(labels))
testdata=features_norm[:,1:len]
testlabels=labels[1:len]
treedata=features_norm[:,len+1:end]
treelabels=labels[len+1:end]
# 创建树
tree=KDTree(treedata)
# k临近搜索，取出和测试点最近的三个点并排序
idxs,dists=knn(tree,testdata,3,true)
string_idx=[treelabels[i] for i in idxs]
# 多数表决
function class(list)
    sub=unique(list)
    if length(sub)==1
        return sub[1]
    else
        result=[count(s->s==c,list) for c in sub]
        return sub[argmax(result)]
    end
end
result=[class(list) for list in string_idx]
result_compare=(result.==testlabels)
corrects=count(result_compare)
errors=len-corrects
@info("约会问题计算完毕，稍候进行数字识别的计算")
sleep(5)

#数字识别计算范例
#先解压digital.zip到当前文件夹，解压完毕将新增两个文件夹。
#为减小存储体积，计算完毕后可以删除两个新增的文件夹。
function img2vector(filename)
    str=read(filename,String)
    strnum=replace(str,"\r\n"=>"")
    substr=split(strnum,"")
    numarr=parse.(Float32,substr)
    return numarr
end

# 读入所有文件，整理特征数据为一个矩阵，一个图像文件是一列。
trainingFileList=readdir("trainingDigits")
data=[img2vector(string("trainingDigits/",f)) for f in trainingFileList]
trainMax=hcat(data...)

# 测试标签数据
trainLabel=first.(trainingFileList)

# 创建训练数据树
tree=KDTree(trainMax)

# 测试数据
testFileList=readdir("testDigits")
data=[img2vector(string("testDigits/",f)) for f in testFileList]
testMax=hcat(data...)

# 利用kNN进行距离计算
idxs,dists=knn(tree,testMax,3,true)
char_idx=[trainLabel[i] for i in idxs]
result=class.(char_idx)

# 和测试数据集的标签数据进行比较
testLabel=first.(testFileList)
corrects=count(result.==testLabel)
errors=length(testLabel)-corrects
error_rate=errors/length(testLabel)