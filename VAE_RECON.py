from mxnet import nd
from mxnet import gluon
from mxnet import init
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

# Upload trained Generator parameters
import VAE as vaemodule
vae = vaemodule.VAE()
filename1 = './params/vae.params.save'
vae.load_params(filename1, ctx = ctx)
print(vae)

#if not updating the seed by system time, you'll get the same results 
import time
from mxnet import random
seed = int(time.time()*100)
random.seed(seed)

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso

####################################################
## Parameter Configuration

noise_std = 0.1
lmbd = 0.1
num_measurements = 300
num_random_restarts = 10  ## better to be large, 10 is OK
n_z = 20


####################################################
## Algorithm Start
## This Demo only use one batch to show the effecti-
## veness of generative model based Compressed Sens-
## ing
batch_size = 10
total_epoch = 1000
train_data, test_data = utils.load_data_mnist(batch_size)
for x_batch, label in test_data:

	# Generate A, Noise and y_batch
	A = nd.random.normal(shape = (784, num_measurements))
	noise_batch = noise_std * nd.random.normal(shape = (batch_size, num_measurements))
	x_batch = x_batch.reshape((batch_size, 784))
	y_batch = nd.dot(x_batch, A) + noise_batch

	########################
	### Lasso
	########################
	x_hat_batch_Lasso = nd.zeros([batch_size, 784])
	lasso_est = Lasso(alpha = lmbd)
	for i in range(batch_size):
		y_val = y_batch[i]
		lasso_est.fit(A.T.asnumpy(), y_val.reshape(num_measurements).asnumpy())
		x_hat_lasso = nd.array(lasso_est.coef_)
		x_hat_lasso = nd.reshape(x_hat_lasso, [-1])
		x_hat_lasso = nd.maximum(nd.minimum(x_hat_lasso, 1), 0)
		x_hat_batch_Lasso[i] = x_hat_lasso

	########################
	### OMP Algorithm
	########################
	omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs = num_measurements/2)
	x_hat_batch_OMP = nd.zeros([batch_size, 784])
	for i in range(batch_size):
		y_val = y_batch[i]
		omp_est.fit(A.T.asnumpy(), y_val.reshape(num_measurements).asnumpy())
		x_hat_OMP = nd.array(omp_est.coef_)
		x_hat_OMP = nd.reshape(x_hat_OMP,[-1])
		x_hat_OMP = nd.maximum(nd.minimum(x_hat_OMP, 1), 0)
		x_hat_batch_OMP[i] = x_hat_OMP

	########################
	### Generative Model
	########################
	# x_recon_batch: the collection of reconstruction of the images
	# x_recon_loss: the corresponding loss for the reconstruction of each image
	# which is used for finding the best reconstruction
	x_recon_batch = nd.zeros((batch_size, 784))
	x_recon_loss = nd.ones((batch_size, ))*10000

	# Use different initialization of z
	for restart in range(num_random_restarts):
		tic = time.time()

		train_last_loss = 2.
		train_curr_loss = 0.1

		# Put z into the dict of parameters to be optimized
		# Only z will be updated in this algorithm
		paramdict = gluon.ParameterDict('noise')
		paramdict.get('z', shape = (batch_size, n_z), init = init.Normal(1)) #default sigma is 0.01
		paramdict.initialize(ctx = ctx)
		z = paramdict.get('z').data()
		trainer = gluon.Trainer(paramdict, 'Adam', {'learning_rate': 0.01})

		# Define Loss
		recon_loss = vaemodule.Recon_Loss()
		z_loss = vaemodule.Z_Loss()

		## Optimization process: find the best z
		for epoch in range(total_epoch):
			if abs(train_last_loss - train_curr_loss)/train_last_loss < 1e-3:
				break

			with autograd.record():
				x_hat_batch = vae.decode(z)
				loss1 = recon_loss(A, y_batch, x_hat_batch)
				loss2 = z_loss(z, lmbd)
				loss = loss1+loss2

			loss.backward()
			trainer.step(batch_size)
			loss1_value = nd.mean(loss1).asscalar()
			loss2_value = nd.mean(loss2).asscalar()

			if epoch%990 == 0:
				print('Epoch %2d, loss1, %f, loss2, %f, time %.1f sec' %(epoch, loss1_value, loss2_value, time.time()-tic))

		## Find the best reconstruction of each image in the batch
		loss_value = nd.array(loss)
		for index in range(batch_size):
			if loss_value[index] < x_recon_loss[index]:
				x_recon_batch[index] = x_hat_batch[index]
				x_recon_loss[index] = loss_value[index]

	break ## Exit after one batch


# Show images
print(x_recon_loss)
images = nd.concat(x_batch, x_hat_batch_Lasso, x_hat_batch_OMP, x_recon_batch, dim = 0)
utils.show_images_2D(images, 4, 10)



