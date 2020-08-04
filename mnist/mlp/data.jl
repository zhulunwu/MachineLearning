using Colors

# mnist数据文件路径
const TRAINIMAGES ="D:\\Explore\\Common\\mnist\\train-images-idx3-ubyte"
const TRAINLABELS ="D:\\Explore\\Common\\mnist\\train-labels-idx1-ubyte"
const TESTIMAGES = "D:\\Explore\\Common\\mnist\\t10k-images-idx3-ubyte"
const TESTLABELS = "D:\\Explore\\Common\\mnist\\t10k-labels-idx1-ubyte"

const Gray = Colors.Gray{Colors.N0f8}

const IMAGEOFFSET = 16
const LABELOFFSET = 8

const NROWS = 28
const NCOLS = 28

function imageheader(io::IO)
  magic_number = bswap(read(io, UInt32))
  total_items = bswap(read(io, UInt32))
  nrows = bswap(read(io, UInt32))
  ncols = bswap(read(io, UInt32))
  return magic_number, Int(total_items), Int(nrows), Int(ncols)
end

function labelheader(io::IO)
  magic_number = bswap(read(io, UInt32))
  total_items = bswap(read(io, UInt32))
  return magic_number, Int(total_items)
end

function rawimage(io::IO)
  img = Array{Gray}(undef, NCOLS, NROWS)
  for i in 1:NCOLS, j in 1:NROWS
    img[i, j] = reinterpret(Colors.N0f8, read(io, UInt8))
  end
  return img
end

function rawimage(io::IO, index::Integer)
  seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
  return rawimage(io)
end

rawlabel(io::IO) = Int(read(io, UInt8))

function rawlabel(io::IO, index::Integer)
  seek(io, LABELOFFSET + (index - 1))
  return rawlabel(io)
end

getfeatures(io::IO, index::Integer) = vec(getimage(io, index))


function minst_images(set = :train)
  io = IOBuffer(read(set == :train ? TRAINIMAGES : TESTIMAGES))
  _, N, nrows, ncols = imageheader(io)
  [rawimage(io) for _ in 1:N]
end

function minst_labels(set = :train)
  io = IOBuffer(read(set == :train ? TRAINLABELS : TESTLABELS))
  _, N = labelheader(io)
  [rawlabel(io) for _ = 1:N]
end

# 用法范例
#=
imgs = minst_images() #返回所有图像
imgs[1]               #显示第一张图片，数据类型是ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}
# 数据转换为float类型的数据
float(imgs[1]) # 得到28*28的矩阵
lbls=minst_labels() # 60000个标签，一维数组。
=#