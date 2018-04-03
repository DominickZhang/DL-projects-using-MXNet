from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import loss
from mxnet import random
from mxnet import autograd
import matplotlib.pyplot as plt
import time
import sys
#sys.path.append('..') Add the upper directory
sys.path.append('./dependencies')
import utils
ctx = utils.try_gpu()

random.seed(1)

def Reparameterize(mu, logvar):
	std = nd.random_normal(0, 1, shape = mu.shape)
	return nd.exp(0.5*logvar)*std+mu

class VAE(gluon.HybridBlock):
	def __init__(self, **kwargs):
		# output: the recoverd images
		# decodeoutput: the generated images from random noise
		# mu: mean value for the mixture Gaussian model
		self.output = None
		self.decodeoutput = None
		self.mu = None

		super(VAE, self).__init__(**kwargs)
		with self.name_scope():
			self.encoder = nn.HybridSequential(prefix='encoder')
			self.encoder.add(nn.Dense(400, activation = "relu"))
			self.encoder.add(nn.Dense(40, activation = None))

			self.decoder = nn.HybridSequential(prefix='decoder')
			self.decoder.add(nn.Dense(400, activation = "relu"))
			self.decoder.add(nn.Dense(784, activation = "sigmoid"))

	def hybrid_forward(self, F, x):
		h = self.encoder(x)
		mu_logvar = F.split(h, axis = 1, num_outputs = 2)
		mu = mu_logvar[0]
		logvar = mu_logvar[1]
		self.mu = mu

		z = Reparameterize(mu, logvar)
		y = self.decoder(z)
		self.output = y

		loss1 = -0.5*F.sum((1+logvar-F.power(mu,2)-F.exp(logvar)), axis = 1)
		loss2 = F.sum((x.reshape(y.shape) - y)*(x.reshape(y.shape) - y), axis = 1)

		loss = loss1 + loss2
		return loss

	def decode(self, z):
		y = self.decoder(z)
		self.decodeoutput = y
		return y