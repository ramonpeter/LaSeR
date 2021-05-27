from utils.train_utils import *
from load_data import *

from gan_models import *
from flow_model import INN

import sys, os

import config_flow as c
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

Flow = INN(num_coupling_layers=c.n_blocks, in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units)
Flow.define_model_architecture()
Flow.set_optimizer()
D = netD(in_dim=data_shape, num_layers=2*c.n_layers, internal_size=2*c.n_units)
D.define_model_architecture_unreg()
D.set_optimizer()

print("\n" + "==="*30 + "\n")
print(Flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in Flow.params_trainable]))
print("\n" + "==="*30 + "\n")
print(D)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "==="*30 + "\n")

criterion_BCE = nn.BCEWithLogitsLoss().to(device)
phi_1 = lambda dreal, lreal, lfake: criterion_BCE(dreal, lreal)
phi_2 = lambda dfake, lreal, lfake: criterion_BCE(dfake, lfake)

try:
	log_dir = c.save_dir

	D_loss_meter = AverageMeter()
	F_loss_meter = AverageMeter()

	'''Load a pre-trained primary generator'''
	checkpoint_path_F = log_dir + '/' +  c.dataset + '/n_epochs_200/' + '/checkpoint_F_epoch_100.pth'
	Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path_F, Flow, Flow.optim)

	Flow.model.eval()

	for epoch in range(c.n_epochs):
		for iteration in range(c.n_its_per_epoch):

			i=0

			for events in train_loader:
				events /= scales

				D.model.train()
				
				D.optim.zero_grad()

				label_real = torch.ones(c.batch_size).double().to(device)
				label_fake = torch.zeros(c.batch_size).double().to(device)
				
				'''Train on real data'''
				d_result_real = D(events).view(-1)
				d_loss_real_ = phi_1(d_result_real, label_real, None).mean(-1)

				noise = torch.randn(c.batch_size, data_shape).double().to(device)
				fake = Flow.model(noise, rev=True).view(c.batch_size, data_shape)				

				'''Train on fake data'''
				d_result_fake = D(fake).view(-1)
				d_loss_fake_ = phi_2(d_result_fake, None, label_fake).mean()
				d_loss = d_loss_real_ + d_loss_fake_

				D_loss_meter.update(d_loss.item())

				d_loss.backward()
				D.optim.step()

				i += 1

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), D.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, D_loss_meter, F_loss_meter)

			elif (epoch + 1) == len(train_loader):
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), D.scheduler.optimizer.param_groups[0]['lr'],
							   (i + 1) % c.show_interval, D_loss_meter, F_loss_meter)

			D_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_D = {
					'epoch': epoch,
					'model': D.model.state_dict(),
					'optimizer': D.optim.state_dict(),
					}
				save_checkpoint(checkpoint_D, log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs), 'checkpoint_D_epoch_%03d' % (epoch))

		D.scheduler.step()

except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 
