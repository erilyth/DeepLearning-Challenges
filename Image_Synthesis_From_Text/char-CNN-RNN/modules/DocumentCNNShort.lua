
local DocumentCNN = {}
function DocumentCNN.cnn(alphasize, emb_dim, dropout)
  dropout = dropout or 0.0

  local net = nn.Sequential()
  -- 201 x alphasize
  net:add(nn.TemporalConvolution(alphasize, 256, 4))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 66 x 256
  net:add(nn.TemporalConvolution(256, 256, 4))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 21 x 256
  net:add(nn.TemporalConvolution(256, 256, 4))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 6 x 256
  net:add(nn.Reshape(1536))
  -- 1536
  net:add(nn.Linear(1536, 1024))
  net:add(nn.Dropout(dropout))
  net:add(nn.Linear(1024, emb_dim))
  return net
end

return DocumentCNN

