import argparse
import config_flow as c

def parse(args):

	parser = argparse.ArgumentParser(prog=args[0])

	parser.add_argument('-l', '--lr',            default=c.lr, dest='lr', type=float)
	parser.add_argument('-d', '--gamma',         default=c.gamma, dest='gamma', type=float)

	parser.add_argument('-b', '--batch_size',    default=c.batch_size, dest='batch_size', type=int)
	parser.add_argument('-n', '--n_iterations',  default=c.n_its_per_epoch, dest='n_its_per_epoch', type=int)
	parser.add_argument('-N', '--epochs',        default=c.n_epochs, dest='n_epochs', type=int)

	opts = parser.parse_args(args[1:])

	c.lr              = opts.lr
	c.batch_size      = opts.batch_size
	c.gamma           = opts.gamma
	c.n_its_per_epoch = opts.n_its_per_epoch
	c.n_epochs        = opts.n_epochs
