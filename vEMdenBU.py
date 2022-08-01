import argparse
import distutils
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Datasets as Datasets
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Denoiser as Denoiser
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Losses as Losses
import FiReTiTiPyLib.IO.IOColors as Colors
import FiReTiTiPyLib.Normalizers as Normalizer
import FiReTiTiPyLib.PyTorch_Models.Denoising as models
import gc
import os
import random
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as Transformer

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


def str2bool(v):
	return bool(distutils.util.strtobool(v))


parser = argparse.ArgumentParser(description='Volume Electron Microscopy Denoiser (vEMden) Example')
parser.add_argument("--dataDir", type=str, default="./Data/", help="Path to the data to denoise.")
parser.add_argument('--batchSize', type=int, default=12, help='Training/Inference batch size. Default=12')
parser.add_argument('--nIterations', type=int, default=1001, help='Minimum number of iteration (weights update) to '
																  + 'perform. Minimum recommended: 500 for NRRN and 1000 for DenoiseNet.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001.')
parser.add_argument('--nThreads', type=int, default=4, help='Number of threads to use by the data loaders.')
parser.add_argument('--seed', type=int, help='Random seed.')
#parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, help='Use cuda?')
#parser.add_argument('--cuda', type=bool, default=True, help='Use cuda? Default=True')
#parser.add_argument('--cuda', action='store_true', help='Use cuda? Default=True')
parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=False, help='Use cuda? Default=False')
#parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
#parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restart)")
parser.add_argument("--net", default="NRRN", type=str, help="net could be: NRRN(default) | DenoiseNet | "
															+ "PathToSavedModel")
parser.add_argument("--nBlocks", default=4, type=int, help="Number of denoising blocsk/layers (model depth)."
														   + " Defaults=4.")
parser.add_argument("--nFeaturesMaps", default=32, type=int, help="Number of features maps (model width). Defaults=32.")
parser.add_argument("--trainingSize", default=-1, type=int, help="Number of pairs/triplets in the dataset used for "
																 + "training. Defaults=-1, which means all images will be used.")
parser.add_argument("--cropSize", default=256, type=int, help="The size of the crops/patches used during training and "
															  + "inference. Defaults=256.")
parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='Use debug mode?')


def _InputTransform_(cropSize):
	"""
		Performs transformation on the input image
	"""
	return Compose([Transformer.RandomCrop(cropSize),
					Transformer.RandomHorizontalFlip(),
					Transformer.RandomVerticalFlip()])


def _TrainSingleEpoch2_(model, optimizer, criterion, device, epoch, DL, verbose: bool = True):
	epoch_loss = 0.0
	tstart = time.time()
	for iteration, batch in enumerate(DL, 1):
		optimizer.zero_grad()
		
		image, imnext = batch[0].to(device), batch[1].to(device)
		
		out = model(image)
		
		loss = criterion(out, imnext)
		
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
	
	aveloss = epoch_loss / len(DL)
	if verbose:
		print("---> Epoch {} Complete: Avg. Loss = {:.5f} in {:.5f}s.".format(epoch, aveloss, time.time() - tstart))
	return aveloss


def _TrainSingleEpoch3_(model, optimizer, criterion, device, epoch, DL, verbose: bool = True):
	epoch_loss = 0.0
	tstart = time.time()
	for iteration, batch in enumerate(DL, 1):
		optimizer.zero_grad()
		
		imprev, image, imnext = batch[0].to(device), batch[1].to(device), batch[2].to(device)
		
		x = torch.cat((imprev, image), dim=1)  # concat the 2 images on the channel dim => (batch,2,256,256)
		y = torch.cat((imnext, image), dim=1)
		
		out_x = model(x)
		out_y = model(y)
		
		loss = criterion(imprev, imnext, out_x, out_y)
		
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
	
	aveloss = epoch_loss / len(DL)
	if verbose:
		print("---> Epoch {} Complete: Avg. Loss = {:.5f} in {:.5f}s.".format(epoch, aveloss, time.time() - tstart))
	return aveloss


def _Run_():
	opt = parser.parse_args()
	if opt.debug:
		print("Debugging mode activated.")
		print("Command line arguments:")
		print(opt)
	
	print(Colors.Colors.GREEN + '\n=====> Checking parameters' + Colors.Colors.RESET)
	if opt.seed is not None:
		random.seed(opt.seed)
		torch.manual_seed(opt.seed)
		cudnn.deterministic = True
		print(Colors.Colors.RED + '\nWARNING - ' + Colors.Colors.RESET + 'You have chosen to seed training. '
																		 'This will turn on the CUDNN deterministic setting, which can slow down your training considerably!\n')
	
	if opt.cuda and not torch.cuda.is_available():
		raise Exception("No GPU found but option '--cuda' enabled, please disable cuda option '--cuda=False' "
						"or run with a GPU.")
	print("Parameters preliminary ckecking done.")
	
	print(Colors.Colors.GREEN + '\n=====> Building model' + Colors.Colors.RESET)
	model, nInputs, ModelType = models.getDenoisinNetwork(NET_TYPE=opt.net, nbBlocks=opt.nBlocks,
														  FeatureMaps=opt.nFeaturesMaps)
	if model is None:
		Training = False
		print("Loading previously trained model... ", end='')
		try:
			#model = torch.load(opt.net, map_location=torch.device("cpu")) # Does not work with DataParallel.
			modelname = os.path.basename(opt.net)
			elements = modelname.split('_')
			nBlocks = int(elements[2][8:len(elements[2])])
			nMaps = int(elements[3][6:len(elements[3])])
			if "nrrn" in opt.net.lower():
				Type = "NRRN"
			elif "denoisenet" in opt.net.lower():
				Type = "DenoiseNet"
			else:
				print(Colors.Colors.RED + '\nERROR - Unknown denoising model type.\n' + Colors.Colors.RESET)
				raise Exception('Unknown denoising model type.')
			model, nInputs, ModelType = models.getDenoisinNetwork(NET_TYPE=Type, nbBlocks=nBlocks, FeatureMaps=nMaps)
			model.load_state_dict(torch.load(opt.net))
			print("succesfully.")
			print("Denoising model type = " + ModelType)
		except:
			print(
				Colors.Colors.RED + "\nERROR - '" + opt.net + "' does not contain a valid model." + Colors.Colors.RESET)
			sys.stdout.flush()
			raise Exception("'" + opt.net + "' does not contain a valid model.")
	else:
		Training = True
		print("succesfully.")
	
	if Training:
		print("The model will be trained before denoising the dataset.")
	else:
		print("Model already trained, so skipping training.")
	
	print(Colors.Colors.GREEN + '\n=====> Checking environment' + Colors.Colors.RESET)
	device = torch.device("cuda" if opt.cuda else "cpu")
	warning = '\n' if str(device) == 'cuda' else Colors.Colors.RED + "\nWARNING - " + Colors.Colors.RESET
	print(warning + "Device = " + str(device) + "\n")
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)
	print("Model successfully linked to environment.")
	print("Environment checked.")
	
	normalizer = Normalizer.CenterReduce(MaxValue=255.0)
	
	if Training:
		print(Colors.Colors.GREEN + '\n=====> Creating DataSet and DataLoader' + Colors.Colors.RESET)
		
		criterion = nn.MSELoss().to(device) if nInputs == 1 else Losses.N2NLoss().to(device)
		TrainSingleEpoch = _TrainSingleEpoch2_ if nInputs == 1 else _TrainSingleEpoch3_
		decay = 0.0001 if nInputs == 1 else 0.0
		optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=decay)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if nInputs == 3 else None
		print("Optimizer, loss and scheduller created.")
		
		path = opt.dataDir
		dataset = Datasets.NRRN(path, normalizer=normalizer, nInputs=nInputs, train=True, datasetSize=opt.trainingSize,
								transformations=_InputTransform_(opt.cropSize), Debug=opt.debug)
		if opt.debug:
			for i in range(dataset.__len__()):  # Quick dataset check.
				if nInputs == 1:
					im, nextim = dataset.__getitem__(i)
				else:
					previm, im, nextim = dataset.__getitem__(i)
		print("Dataset created. Size = " + str(dataset.__len__()))
		
		TrainingDataLoader = DataLoader(dataset=dataset, num_workers=opt.nThreads, batch_size=opt.batchSize,
										drop_last=True, shuffle=True)
		print("DataLoader created.")
		
		print(Colors.Colors.GREEN + '\n=====> Training' + Colors.Colors.RESET)
		nEpochs = int(opt.nIterations / int(dataset.__len__() / opt.batchSize)) + 1
		print("Training configuration:")
		print(" - Minimum " + str(opt.nIterations) + " iterations/updates requested")
		print(" - Batch size = " + str(opt.batchSize))
		print(" ---> Number of epochs = " + str(nEpochs))
		print(" ---> %d iterations per epoch" % (int(dataset.__len__() / opt.batchSize)))
		print(" - Crop size = " + str(opt.cropSize))
		print(" - Learning rate = " + str(opt.lr))
		print(" - Cuda = " + str(True if opt.cuda else False))
		print(" - Number of input image(s) = " + str(nInputs))
		
		for epoch in range(0, nEpochs):
			TrainSingleEpoch(model, optimizer, criterion, device, epoch, TrainingDataLoader)
			if scheduler is not None:
				scheduler.step()
		print("Training done.")
		
		year, month, day, hour, minutes = map(int, time.strftime("%Y %m %d %H %M").split())
		name = "Denoising_" + ModelType + "_nBlocks=" + str(opt.nBlocks) + "_nMaps=" + str(opt.nFeaturesMaps) + \
			   "_Date=" + str(year) + "." + str(month) + "." + str(day) + "." + str(hour) + "h" + str(minutes) + ".pt"
		#torch.save(model, name)
		if torch.cuda.device_count() > 1:
			torch.save(model.module.state_dict(), name)
		else:
			torch.save(model.state_dict(), name)
		print("Model saved as: " + name)
		sys.stdout.flush()
		
		del criterion
		del scheduler
		del optimizer
		del TrainingDataLoader
		del dataset
		if device == "cuda":
			torch.cuda.empty_cache()
		gc.collect()
	
	print(Colors.Colors.GREEN + '\n=====> Densoising / Inference' + Colors.Colors.RESET)
	
	model.eval()
	
	print("Densoising / Inference configuration:")
	print(" - Batch size = " + str(opt.batchSize))
	print(" - Crop size = " + str(opt.cropSize))
	print(" - Cuda = " + str(True if opt.cuda else False))
	print(" - Number of input image(s) = " + str(nInputs))
	sys.stdout.flush()
	
	path = opt.dataDir
	if path[len(path) - 1] == '/':
		path = path[0:len(path) - 1]
	
	dataset = Datasets.NRRN(path, nInputs=nInputs, train=False, Debug=opt.debug)
	
	denoiser = Denoiser.Denoiser(verbose=False)
	denoiser.Denoise(dataset, model, nInputs=nInputs, CropSize=opt.cropSize, BorderEffectSize=opt.nBlocks + 5,
					 BatchSize=opt.batchSize, Device=device, Normalizer=normalizer,
					 ResultsDirPath=path + " - Denoised " + ModelType)
	
	print(Colors.Colors.GREEN + "\nAll Done." + Colors.Colors.RESET)


def Denoise(parameters=None):
	if parameters is not None:
		parser.set_defaults(**parameters)
	_Run_()


if __name__ == '__main__':
	parameters = {'dataDir': "/Users/firetiti/Downloads/EM/Denoising/Example/Registered Crop 1024x1024/",
				  'batchSize': 6,
				  'nIterations': 23,
				  'lr': 0.0001,
				  'nThreads': 2,
				  #'seed': 13,
				  'cuda': False,
				  'net': 'NRrn',
				  'nBlocks': 2,
				  'nFeaturesMaps': 16,
				  'trainingSize': -1,
				  #'cropSize': 256,
				  'debug': True}
	Denoise(parameters=parameters)
#Denoise()
