from utils.train_utils import *
from utils.observables import *
from utils.plotting.distributions import *
from utils.plotting.plots import *
from load_data import *

from gan_models import *

import sys, os

import config_gan as c
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

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.weighted)

if c.weighted:
	data_shape -= 1

G = netG(in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units)
G.define_model_architecture()
G.set_optimizer()

D = netD(in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units)
D.define_model_architecture_unreg()
D.set_optimizer()

data_shape *= c.latent_dim_gen

print("\n" + "==="*30 + "\n")
print(G)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in G.params_trainable]))
print("\n" + "==="*30 + "\n")
print(D)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "==="*30 + "\n")

criterion_BCE = nn.BCEWithLogitsLoss().to(device)
phi_1 = lambda dreal, lreal, lfake: criterion_BCE(dreal, lreal)
phi_2 = lambda dfake, lreal, lfake: criterion_BCE(dfake, lfake)
phi_3 = lambda dfake, lreal, lfake: criterion_BCE(dfake, lreal)

try:
	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/' +  c.dataset + '/' + '/n_epochs_' + str(c.n_epochs))

	# setup some varibles
	G_loss_meter = AverageMeter()
	D_loss_meter = AverageMeter()

	G_loss_list = []
	D_loss_list = []

	for epoch in range(c.n_epochs):
		for iteration in range(c.n_its_per_epoch):

			i=0
			j=0

			for data in train_loader:

				if c.weighted:
					events  = data[:,:-1]
					weights = data[:,-1:] / 1.
				else:					
					events = data / scales

				G.model.train()
				D.model.train()
				G.optim.zero_grad()
				D.optim.zero_grad()

				if c.train:
					'''Train discriminator'''
					for nd in range(c.n_disc_updates):
						label_real = torch.ones(c.batch_size).double().to(device)
						label_fake = torch.zeros(c.batch_size).double().to(device)
					
						d_result_real = D(events).view(-1)

						if c.weighted:
							criterion = nn.BCEWithLogitsLoss(weight=weights.view(-1),reduction='sum')
							d_loss_real_ = criterion(d_result_real, label_real) / torch.sum(weights)

						else:	
							d_loss_real_ = phi_1(d_result_real, label_real, None).mean(-1)

						noise = torch.randn(c.batch_size, data_shape).double().to(device)
						fake = G(noise)				

						d_result_fake = D(fake.detach()).view(-1)
						d_loss_fake_ = phi_2(d_result_fake, None, label_fake).mean()
						d_loss = d_loss_real_ + d_loss_fake_

						D_loss_meter.update(d_loss.item())

						d_loss.backward()
						D.optim.step()

					'''Train generator'''
					noise = torch.randn(c.batch_size, data_shape).double().to(device)
					fake = G(noise)
					d_result_fake = D(fake).view(-1)
					g_loss = phi_3(d_result_fake, label_real, label_fake).mean(-1)

					G_loss_meter.update(g_loss.item())

					g_loss.backward()
					G.optim.step()

				i += 1

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), D.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, D_loss_meter, G_loss_meter, Flow=False)

			elif (epoch + 1) == len(train_loader):
				print_log(epoch, c.n_epochs, i + 1, len(train_loader), D.scheduler.optimizer.param_groups[0]['lr'],
							   (i + 1) % c.show_interval, D_loss_meter, G_loss_meter, Flow=False)

			G_loss_meter.reset()
			D_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_G = {
					'epoch': epoch,
					'model': G.model.state_dict(),
					'optimizer': G.optim.state_dict(),
					}
				save_checkpoint(checkpoint_G, log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs), 'checkpoint_G_epoch_%03d' % (epoch))

				checkpoint_D = {
					'epoch': epoch,
					'model': D.model.state_dict(),
					'optimizer': D.optim.state_dict(),
					}
				save_checkpoint(checkpoint_D, log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs), 'checkpoint_D_epoch_%03d' % (epoch))

			if c.test == True:
				size = 1000
			else:
				size = 100000

			with torch.no_grad():
				real = get_real_data(c.dataset, c.test, size)

				noise = torch.randn(size, data_shape).double().to(device)
				if c.weighted:
					generated = G(noise).detach().numpy()
				else:
					generated = G(noise).detach().numpy() * scales

			#distributions = Distribution(real, generated, 'epoch_%03d' % (epoch) + '_target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset)
			distributions = Distribution(real, generated, 'epoch_%03d' % (epoch) + '_target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, weights = real[:,-1], latent=True)
			distributions.plot()

		G.scheduler.step()
		D.scheduler.step()

except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 
