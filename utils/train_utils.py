import torch

import pandas as pd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPSILON = 1.e-8


# ======================================================================================================================
class AverageMeter(object):
	""" Computes ans stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def print_log(epoch, epoches, iteration, iters, learning_rate, display, loss_D, loss_G, Flow):
	if Flow:
		print('epoch: [{}/{}] iteration: [{}/{}]\t' 'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
		print('Loss_F = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'.format(display, loss_D=loss_D))
	else:
		print('epoch: [{}/{}] iteration: [{}/{}]\t' 'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
		print('Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})'.format(display, loss_D=loss_D))
		print('Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'.format(display, loss_G=loss_G))
	

# ======================================================================================================================

def save_checkpoint(state, log_dir, name):
	path = log_dir + '/' + name + '.pth'
	torch.save(state, path)

def load_checkpoint(path, model, optimizer):
	checkpoint = torch.load(path, map_location=torch.device('cpu'))
	model.model.load_state_dict(checkpoint['model'])
	model.optim.load_state_dict(checkpoint['optimizer'])

	return model, optimizer, checkpoint['epoch']

def get_real_data(dataset, test, sample_size):

	datapath = './data/'
	data = pd.read_hdf(datapath + dataset + '.h5').values

	if test == True:
		split = int(len(data) * 0.1)
	else:
		split = int(len(data) * 0.8)

	data = data[:split]

	index = np.arange(data.shape[0])

	if sample_size <= data.shape[0]:
		choice = np.random.choice(index, sample_size, replace=False)
	else:
		choice = np.random.choice(index, sample_size, replace=True)

	data = data[choice]

	return data
