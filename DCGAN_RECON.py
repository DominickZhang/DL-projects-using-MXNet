from mxnet import nd
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon import loss
from mxnet import autograd
import matplotlib.pyplot as plt
import time
import sys
import os
import mxnet as mx
import numpy as np
#sys.path.append('..') Add the upper directory
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()

# Upload trained Generator parameters
import DCGAN as dcgan
filename1 = './params/dcgan.netG.save'
netG = dcgan.Generator()
netG.load_params(filename1, ctx = ctx)

# If not updating the seed by system time, you'll get the same results 
import time
from mxnet import random
seed = int(time.time()*100)
random.seed(seed)

# Image Preprocessing
def transform(data):
	data = mx.image.imresize(data, 64, 64) # state size: (64, 64, 3)
	data = nd.transpose(data, (2, 0, 1))
	data = data.astype(np.float32)/127.5 - 1 # normalize to [-1, 1]
	if data.shape[0] == 1:
		data = nd.tile(data, (3, 1, 1)) # if image is greyscale, repeat 3 times to get RGB image
	return data.reshape((1, ) + data.shape)

def visualize(img_arr):
	plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.axis('off')


#########################################################
## Parameter Configuration

noise_std = 0.01
lmbd = 0.001
num_measurements = 500
n_z = 100
learn_rate = 0.1
num_random_restarts = 10  ## better to be large, 10 is OK
total_epoch = 100


########################################################
## Algorithm Start
## This Demo only use one batch to show the effecti-
## veness of generative model based Compressed Sens-
## ing
batch_size = 3
n_input = 64*64*3

# Read the images in the folder of "images"
data_path = './images'
img_list = []
for path, _, fnames in os.walk(data_path):
	for fname in fnames:
		if not fname.endswith('.jpg'):
			continue
		img = os.path.join(path, fname)
		img_arr = mx.image.imread(img)
		img_arr = transform(img_arr)
		img_list.append(img_arr)
train_data = mx.io.NDArrayIter(data = nd.concatenate(img_list), batch_size = batch_size)

for data_batch in train_data:
	data = data_batch.data[0].as_in_context(ctx)

	# Generate A, Noise and y_batch
	A = nd.random.normal(shape = (n_input, num_measurements))
	noise_batch = noise_std * nd.random.normal(shape = (batch_size, num_measurements))
	x_batch = data.reshape((batch_size, n_input))
	y_batch = nd.dot(x_batch, A) + noise_batch


	########################
	### Generative Model
	########################
	# x_recon_batch: the collection of reconstruction of the images
	# x_recon_loss: the corresponding loss for the reconstruction of each image
	# which is used for finding the best reconstruction
	x_recon_batch = nd.zeros((batch_size, 3, 64, 64))
	x_recon_loss = nd.ones((batch_size, ))*100000

	# Use different initialization of z
	for restart in range(num_random_restarts):
		tic = time.time()
#		
		train_last_loss = 2.
		train_curr_loss = 0.1

		# Put z into the dict of parameters to be optimized
		# Only z will be updated in this algorithm
		paramdict = gluon.ParameterDict('noise')
		paramdict.get('z', shape = (batch_size, n_z, 1, 1), init = init.Normal(1)) #default sigma is 0.01
		paramdict.initialize(ctx = ctx)
		z = paramdict.get('z').data()
		trainer = gluon.Trainer(paramdict, 'Adam', {'learning_rate': learn_rate})

		# Define Loss
		recon_loss = dcgan.Recon_Loss()
		z_loss = dcgan.Z_Loss()

		## Optimization process: find the best z
		for epoch in range(total_epoch):
			if abs(train_last_loss - train_curr_loss)/train_last_loss < 1e-3:
				break

			with autograd.record():
				x_hat_batch = netG(z)
				loss1 = recon_loss(A, y_batch, x_hat_batch.reshape((batch_size, n_input)))
				loss2 = z_loss(z, lmbd)
				loss = loss1+loss2

			loss.backward()
			trainer.step(batch_size)
			loss1_value = nd.mean(loss1).asscalar()
			loss2_value = nd.mean(loss2).asscalar()
		
			if epoch%(total_epoch - 1) == 0:
				print('Epoch %2d, loss1, %f, loss2, %f, time %.1f sec' %(epoch, loss1_value, loss2_value, time.time()-tic))

		## Find the best reconstruction of each image in the batch
		loss_value = nd.array(loss)
		for index in range(batch_size):
			if loss_value[index] < x_recon_loss[index]:
				x_recon_batch[index] = x_hat_batch[index]
				x_recon_loss[index] = loss_value[index]

	break ## Exit after one batch


# Show images
images = nd.concat(data, x_recon_batch, dim = 0)
print(x_recon_loss)
for i in range(2):
	for j in range(batch_size):
		n = i*batch_size+j
		plt.subplot(2, batch_size, n+1)
		plt.imshow(((images[n].asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
		plt.axis('off')
plt.show()






