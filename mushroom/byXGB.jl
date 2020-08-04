using XGBoost

function readlibsvm(fname::String, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, parse(Int, itm[1]) + 1] = parse(Int, itm[2])
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end

train_X, train_Y = readlibsvm("D:/Explore/DataSets/mushroom/agaricus.txt.train",(6513, 126))
test_X, test_Y = readlibsvm("D:/Explore/DataSets/mushroom/agaricus.txt.test", (1611, 126))

num_round = 2
bst = xgboost(train_X, num_round, label = train_Y, eta = 1, max_depth = 2)

pred = predict(bst, test_X)
print("test-error=", sum((pred .> 0.5) .!= test_Y) / float(size(pred)[1]), "\n")

nfold = 5
param = ["max_depth" => 2,"eta" => 1,"objective" => "binary:logistic"]
metrics = ["auc"]
nfold_cv(train_X, num_round, nfold, label = train_Y, param = param, metrics = metrics)