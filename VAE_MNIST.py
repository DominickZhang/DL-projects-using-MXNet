from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import loss
from mxnet import autograd
import matplotlib.pyplot as plt
import time
import sys
#sys.path.append('..') Add the upper directory
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()


import VAE as vaemodule
vae = vaemodule.VAE()
filename1 = './params/vae.params.get'
#vae.load_params(filename1, ctx = ctx)
vae.collect_params()
print(vae)
vae.initialize(ctx = ctx)


batch_size = 128
total_epoch = 100
epoch_size = 500
train_data, test_data = utils.load_data_mnist(batch_size)

trainer = gluon.Trainer(vae.collect_params(), 'sgd', {'learning_rate': 0.01})

train_last_loss = 2.
train_curr_loss = 0.1
for epoch in range(total_epoch):
	if abs(train_last_loss - train_curr_loss)/train_last_loss < 1e-3:
		break
	train_loss = 0.
	tic = time.time()
	num = 0
	for data, label in train_data:
		num+=1
		# Control the number of training data in each epoch
		if num > epoch_size:
			break

		with autograd.record():
			loss = vae(data)
		loss.backward()
		trainer.step(batch_size)
		train_loss += nd.mean(loss).asscalar()
		
	print('Epoch %2d, loss, %f, time %.1f sec' %(epoch, train_loss/num, time.time()-tic))
	train_last_loss = train_curr_loss
	train_curr_loss = train_loss/num

# Show images
utils.show_images_2D(vae.output[0:64],8,8)
utils.show_images_2D(data[0:64],8,8)
filename2 = './params/vae.params.save'
vae.save_params(filename2)



