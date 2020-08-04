using Flux
using Flux: onehot, chunk, batchseq, crossentropy
using StatsBase: wsample
using Base.Iterators: partition

text = collect(String(read("input.txt")))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50 # 序列长度
nbatch = 50 # 批的数量

# chunk(text, nbatch)将所有字符分成50份，序列长度为50，每份长度为length(text)/50
# batchseq 将50份文本转换成序列形式。序列长度为length(text)/50，序列元素含nbatch个字符，每个字符是长度为68的向量。
# partition 将上述序列再分成较短的序列（seqlen）。
# 总而言之，Xs是有小序列(seqlen)构成的长序列(length(text)/50）。
Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

m = Chain(GRU(N, 128), GRU(128, 128), Dense(128, N),softmax)

function loss(xs, ys)
  l = sum(crossentropy.(m.(xs), ys))
  # rnn输入为短序列，为避免梯度计算历史过长，这里进行了截断。
  Flux.truncate!(m)
  return l
end

opt = ADAM(0.01)
tx, ty = (Xs[5], Ys[5]) # 5随便选的测试数据？

Flux.train!(loss, params(m), zip(Xs, Ys), opt,cb = () -> @show loss(tx, ty)) 

# 以下是进行采样
function sample(m, alphabet, len; temp = 1)
  m = cpu(m)
  Flux.reset!(m) #将模型的状态恢复为初始状态？原先是批处理，现在只有一个字符输入。
  buf = IOBuffer()
  c = rand(alphabet) # 随机从字母表中找出一个字符。
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)).data) # m预测概率，根据概率从字母表中选出字符
  end
  return String(take!(buf))
end

sample(m, alphabet, 1000) |> println