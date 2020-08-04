using Base.Iterators

function loadDataSet()
    postingList=[["my", "dog", "has", "flea", "problems", "help", "please"],
                 ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                 ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                 ["stop", "posting", "stupid", "worthless", "garbage"],
                 ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
                 ["quit", "buying", "worthless", "dog", "food", "stupid"]]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
end

# 创建单词表，显然，每个词出现一次即可。
function createVocabList(dataSet)
   return unique(flatten(dataSet))
end

# 将输入词集转为数字向量。
function setOfWords2Vec(vocabList, inputSet)
    retvec=zeros(Int,length(vocabList))
    idx=unique(indexin(inputSet,vocabList))
    idx=filter(!isnothing,idx)
    [retvec[i]=1 for i in idx]
    return retvec
end

function trainNB0(trainMatrix,trainCategory)
    numTrainDocs = size(trainMatrix)[2]
    numWords = size(trainMatrix)[1]
    pAbusive = sum(trainCategory)/float(numTrainDocs) #先验概率=恶意文档/总文档
    p0Num = ones(numWords)
    p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i=1:numTrainDocs
        if trainCategory[i] == 1  # 条件概率：确定文档类型前提下各个词出现的概率
            p1Num+=trainMatrix[:,i] #每个词统计
            p1Denom += sum(trainMatrix[:,i]) #所有词数量统计
        else
            p0Num+=trainMatrix[:,i]
            p0Denom += sum(trainMatrix[:,i])
        end
    end
    p1Vect = log.(p1Num/p1Denom) #P(wi|c)列表，不过是以log形式出现的。
    p0Vect = log.(p0Num/p0Denom) 
    return p0Vect,p1Vect,pAbusive
end

function classifyNB(vec2Classify, p0Vec, p1Vec, pClass1)
    # p(c|w)=p(w|c)*p(c)，取log后变成加法
    # p(w|c)是测试post的条件概率，p1Vec是各个词的条件概率。测试post是个bool矢量，两者相乘即得到p(w|c)
    p1 = sum(vec2Classify .* p1Vec) + log(pClass1)   
    p0 = sum(vec2Classify .* p0Vec) + log(1.0 - pClass1)
    if p1 > p0
        return 1
    else
        return 0
    end
end

function testingNB()
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=hcat([setOfWords2Vec(myVocabList,post) for post in listOPosts]...)
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    testEntry = ["love", "my", "dalmation"]
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    println(testEntry,"classified as: ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ["stupid", "garbage"]
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    println(testEntry,"classified as: ",classifyNB(thisDoc,p0V,p1V,pAb))
end