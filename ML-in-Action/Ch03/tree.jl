# 鱼类识别数据
function  createDataSet()
    dataSet = hcat([1, 1, "yes"], 
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"])
    labels = ["no surfacing","flippers"]
    return dataSet, labels
end

function  calcShannonEnt(dataSet)
    featVec=dataSet[end,:]
    uf=unique(featVec)
    nf=[count(featVec.==f) for f in uf]
    p=nf./length(featVec)
    shannonEnt=-sum(p.*log2.(p))
    return shannonEnt
end

# 数据集分割。axis为某个特征，value为该特征的值。
# 该函数返回该特征为某个值的样本子集。
# 该子集不再包含该特征。
function splitDataSet(dataSet, axis, value)
    idx=dataSet[axis,:].==value
    sub=vcat(dataSet[1:axis-1,idx],dataSet[axis+1:end,idx])
    return sub
end

# 选择最好的特征以分割数据集
function chooseBestFeatureToSplit(dataSet)
    baseEntropy = calcShannonEnt(dataSet) 
    numFeatures = size(dataSet)[1]-1 # 特征的数量，最后为标签。    
    bestInfoGain = 0.0; bestFeature = -1 # 初始化
    for i=1:numFeatures    #iterate over all the features
        uniqueVals=unique(dataSet[i,:])
        newEntropy = 0.0
        for value in uniqueVals
            subDataSet = splitDataSet(dataSet, i, value)
            prob = last(size(subDataSet))/last(size(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        end     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain)       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
        end
    end
    return bestFeature                      #returns an integer
end

function majorityCnt(classList)
    fv=unique(classList)
    return fv[argmax([count(classList.==v) for v in fv])]
end


function createTree(dataSet,labels_input)
    labels=copy(labels_input)
    classList = dataSet[end,:]
    if count(first(classList).==classList)==length(classList) 
        return first(classList) #所有类别都相等，不必分了
    end
    if first(size(dataSet)) == 1 #只有一个特征，或者说特征向量维度为1，无特征可选。
        return majorityCnt(classList)
    end
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = Dict([(bestFeatLabel,Dict())])
    deleteat!(labels,bestFeat)
    featValues = dataSet[bestFeat,:]
    uniqueVals = unique(featValues)
    for value in uniqueVals
        subLabels = copy(labels)       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    end
    return myTree  
end                          
    
function  classify(inputTree,featLabels,testVec)
    firstStr =first(collect(keys(inputTree)))
    secondDict = inputTree[firstStr]
    key =first(testVec[featLabels.==firstStr])
    valueOfFeat = secondDict[key]
    if typeof(valueOfFeat) <: Dict
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else classLabel = valueOfFeat
    end
    return classLabel
end

# 树的创建和使用
dataSet,labels=createDataSet()
tree=createTree(dataSet,labels)
classify(tree,labels,[1,0])
# 使用julia包进行计算
using DecisionTree
dataSet,labels=createDataSet()
input=Int.(dataSet[1:2,:]')
label=string.(dataSet[end,:])
model = DecisionTreeClassifier()
fit!(model,input,label)
class=predict(model,[1 0])