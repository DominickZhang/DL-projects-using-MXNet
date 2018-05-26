from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn


########################################
## Loss Func for Compressive Sensing  ##
########################################
class Recon_Loss(gluon.loss.Loss):
	def __init__(self, batch_axis = 0, **kwargs):
		super(Recon_Loss, self).__init__(None, batch_axis, **kwargs)

	def hybrid_forward(self, F, A, y_batch, x_hat_batch):
		loss = F.power(F.dot(x_hat_batch, A) - y_batch,2)
		return F.mean(loss, axis = 1)

class Z_Loss(gluon.loss.Loss):
	def __init__(self, batch_axis = 0, **kwargs):
		super(Z_Loss, self).__init__(None, batch_axis, **kwargs)

	def hybrid_forward(self, F, z, lmbd):
		z = z.squeeze()
		loss = lmbd*F.power(z,2)
		return F.sum(loss, axis = 1)

########################################
## Classical model in Alec etc's paper##
########################################
class Generator(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(Generator, self).__init__(**kwargs)
		with self.name_scope():
			self.generator = nn.HybridSequential(prefix = 'generator')
			# input is Z, going into a convolution
			self.generator.add(nn.Conv2DTranspose(64 * 8, 4, 1, 0, use_bias = False))
			self.generator.add(nn.BatchNorm())
			self.generator.add(nn.Activation('relu'))
			# state size (64*8)x4x4
			self.generator.add(nn.Conv2DTranspose(64 * 4, 4, 2, 1, use_bias = False))
			self.generator.add(nn.BatchNorm())
			self.generator.add(nn.Activation('relu'))
			# state size (64*4)x8x8
			self.generator.add(nn.Conv2DTranspose(64 * 2, 4, 2, 1, use_bias = False))
			self.generator.add(nn.BatchNorm())
			self.generator.add(nn.Activation('relu'))
			# state size (64*2)x16x16
			self.generator.add(nn.Conv2DTranspose(64 * 1, 4, 2, 1, use_bias = False))
			self.generator.add(nn.BatchNorm())
			self.generator.add(nn.Activation('relu'))
			# state size (64*1)x32x32
			self.generator.add(nn.Conv2DTranspose(3, 4, 2, 1, use_bias = False))
			self.generator.add(nn.Activation('tanh'))
			# state size 3x64x64
	def hybrid_forward(self, F, x):
		return self.generator(x)

class Discriminator(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(Discriminator, self).__init__(**kwargs)
		with self.name_scope():
			self.discriminator = nn.HybridSequential(prefix = 'discriminator')
			# input is an image, state size 3x64x64
			self.discriminator.add(nn.Conv2D(64, 4, 2, 1, use_bias = False))
			self.discriminator.add(nn.LeakyReLU(0.2))
			# state size (64*1)x32x32
			self.discriminator.add(nn.Conv2D(64 * 2, 4, 2, 1, use_bias = False))
			self.discriminator.add(nn.BatchNorm())
			self.discriminator.add(nn.LeakyReLU(0.2))
			# state size (64*2)x16x16
			self.discriminator.add(nn.Conv2D(64 * 4, 4, 2, 1, use_bias = False))
			self.discriminator.add(nn.BatchNorm())
			self.discriminator.add(nn.LeakyReLU(0.2))
			# state size (64*4)x8x8
			self.discriminator.add(nn.Conv2D(64 * 8, 4, 2, 1, use_bias = False))
			self.discriminator.add(nn.BatchNorm())
			self.discriminator.add(nn.LeakyReLU(0.2))
			# state size (64*8)x4x4
			self.discriminator.add(nn.Conv2D(1, 4, 1, 0, use_bias = False))
			# state size 1*1*1
			# the output is a real number 
			# since there is no sigmoid layer here, we use SigmoidBinaryCrossEntropy as the loss function

	def hybrid_forward(self, F, x):
		return self.discriminator(x)
