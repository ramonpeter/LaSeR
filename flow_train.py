from utils.train_utils import *
from utils.plotting.distributions import *
from utils.plotting.plots import *
from load_data import *

from flow_model import INN

import sys, os

import config_FLOW as c
import opts
opts.parse(sys.argv)
config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
    if v[0]=='_': continue
    s=eval('c.%s'%(v))
    config_str += " {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.on_shell, c.mom_cons, c.weighted)

if c.weighted:
	data_shape -= 1

Flow = INN(num_coupling_layers=c.n_blocks, in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units)
Flow.define_model_architecture()
Flow.set_optimizer()

print("\n" + "==="*30 + "\n")
print(Flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in Flow.params_trainable]))
print("\n" + "==="*30 + "\n")

try:
	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/' +  c.dataset + '/' + '/n_epochs_' + str(c.n_epochs))

	F_loss_meter = AverageMeter()

	if c.load_model:
		checkpoint_path_F = log_dir + '/' + c.dataset + '/n_epochs_200/' + '/checkpoint_F_epoch_100.pth'
		Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path_F, Flow, Flow.optim)

	for epoch in range(c.n_epochs):
		for iteration in range(c.n_its_per_epoch):

			i=0

			for data in train_loader:

				Flow.model.train()			
				Flow.optim.zero_grad()

				if c.weighted:
					events  = data[:,:-1]
					weights = data[:,-1] 

					gauss_output = Flow.model(events.double())
					temp = torch.sum(gauss_output**2/2,1)

					f_loss = torch.mean(weights * temp) - torch.mean(weights * Flow.model.log_jacobian(run_forward=False))

				else:
					events = data / scales

					gauss_output = Flow.model(events.double())
					f_loss = torch.mean(gauss_output**2/2) - torch.mean(Flow.model.log_jacobian(run_forward=False)) / gauss_output.shape[1]

				F_loss_meter.update(f_loss.item())

				f_loss.backward()
				Flow.optim.step()

				i += 1

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), Flow.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, F_loss_meter, F_loss_meter)

			elif (epoch + 1) == len(train_loader):
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), Flow.scheduler.optimizer.param_groups[0]['lr'],
							   (i + 1) % c.show_interval, F_loss_meter, F_loss_meter)

			F_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_F = {
					'epoch': epoch,
					'model': Flow.model.state_dict(),
					'optimizer': Flow.optim.state_dict(),
					}
				save_checkpoint(checkpoint_F, log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), 'checkpoint_F_epoch_%03d' % (epoch))

			if c.test == True:
				size = 1000
			else:
				size = 100000

			with torch.no_grad():
				real = get_real_data(c.dataset, c.test, size)
				noise = torch.randn(size, data_shape).double().to(device)

				if c.weighted:
					inv = Flow.model(noise, rev=True).detach().numpy().reshape(size,data_shape)
				else:
					inv = Flow.model(noise, rev=True).detach().numpy().reshape(size,data_shape) * scales

			distributions = Distribution(real, inv, 'epoch_%03d' % (epoch) + '_target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset)
			distributions.plot()

		Flow.scheduler.step()

except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 
