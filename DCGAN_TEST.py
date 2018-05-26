import DCGAN as dcgan
from mxnet import nd
from mxnet import random
from matplotlib import pyplot as plt
import numpy as np

#if not updating the seed by system time, you'll get the same results 
import time
seed = int(time.time()*100)
random.seed(seed)

import sys
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()

filename = './params/dcgan.netG.save'
netG = dcgan.Generator()
netG.collect_params()
netG.load_params(filename, ctx = ctx)


z = nd.random_normal(0, 1, shape=(4, 100, 1, 1), ctx=ctx)
#print(z)
output = netG(z)

for i in range(4):
	plt.subplot(1, 4, i+1)
	plt.imshow(((output[i].asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.axis('off')
plt.show()

