import argparse
import distutils
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Datasets as Datasets
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Denoiser as Denoiser
import FiReTiTiPyLib.FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Losses as Losses
import FiReTiTiPyLib.IO.IOColors as Colors
import FiReTiTiPyLib.Normalizers as Normalizer
import FiReTiTiPyLib.PyTorch_Models.Denoising as models
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
parser.add_argument("--dataDir", type=str, required=True, help="Path to the data to denoise.")
parser.add_argument('--batchSize', type=int, default=16, help='Training batch size')
parser.add_argument('--nIterations', type=int, default=10001, help='Minimum number of iteration (weights update) to perform.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001.')
parser.add_argument('--nThreads', type=int, default=4, help='Number of threads to use by the data loaders.')
#parser.add_argument('--seed', type=int, default=13, help='random seed to use. Default=13')
parser.add_argument('--seed', type=int, help='Random seed.')
#parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, help='Use cuda?')
#parser.add_argument('--cuda', type=bool, default=True, help='Use cuda? Default=True')
#parser.add_argument('--cuda', action='store_true', help='Use cuda? Default=True')
parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=False, help='Use cuda? Default=False')
#parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
#parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restart)")
#parser.add_argument("--dataset", default="OHSU", type=str, help="dataset could be: EPFL,OHSU (default),TEST,OTH ")
parser.add_argument("--net", default="NRRN", type=str, help="net could be: NRRN(default) | DenoiseNet | PathToSavedModel")
parser.add_argument("--nBlocks", default=5, type=int, help="Number of denoising blocsk/layers (model depth). Defaults=5.")
parser.add_argument("--nFeaturesMaps", default=64, type=int, help="Number of features maps (model width). Defaults=64.")
parser.add_argument("--trainingSize", default=-1, type=int, help="Number of pairs/triplets in the dataset used for training. Defaults=-1, which means all images will be used.")
parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='Use debug mode?')



def _InputTransform_():
	"""
		Performs transformation on the input image
	"""
	return Compose([Transformer.RandomCrop(256), Transformer.RandomHorizontalFlip(), Transformer.RandomVerticalFlip()])


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
		print("---> Epoch {} Complete: Avg. Loss: {:.5f}, {} iterations in {:.5f}s.".format(epoch, aveloss, len(DL),
																							time.time() - tstart))
	return aveloss


def _TrainSingleEpoch3_(model, optimizer, criterion, device, epoch, DL, verbose: bool=True):
	epoch_loss = 0.0
	tstart = time.time()
	for iteration, batch in enumerate(DL, 1):
		optimizer.zero_grad()
		
		imprev, image, imnext = batch[0].to(device), batch[1].to(device), batch[2].to(device)
		
		x = torch.cat((imprev, image), dim=1) # concat the 2 images on the channel dim => (batch,2,256,256)
		y = torch.cat((imnext, image), dim=1)

		out_x = model(x)
		out_y = model(y)

		loss = criterion(imprev, imnext, out_x, out_y)

		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
		
	aveloss = epoch_loss / len(DL)
	if verbose:
		print("---> Epoch {} Complete: Avg. Loss: {:.5f}, {} iterations in {:.5f}s.".format(epoch, aveloss, len(DL),
																							time.time()-tstart))
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
			model = torch.load(opt.net)
			print("succesfully.")
			if "nrrn" in opt.net.lower():
				ModelType = "NRRN"
				nInputs = 3
			elif "denoisenet" in opt.net.lower():
				ModelType = "DenoiseNet"
				nInputs = 1
			else:
				ModelType = "Unknown"
				nInputs = 3
				print(Colors.Colors.RED + '\nWARNING - ' + Colors.Colors.RESET +
					'Unknown denoising model type. Might not work properly.\n')
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
	criterion = nn.MSELoss().to(device) if nInputs == 1 else Losses.N2NLoss().to(device)
	TrainSingleEpoch = _TrainSingleEpoch2_ if nInputs == 1 else _TrainSingleEpoch3_
	optimizer = optim.Adam(model.parameters(), lr=opt.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if nInputs == 3 else None
	print("Environment checked.")
	
	
	if Training:
		print(Colors.Colors.GREEN + '\n=====> Creating DataSet and DataLoader' + Colors.Colors.RESET)
		path = opt.dataDir
		#path = "/Users/firetiti/Downloads/EM/Denoising/101/Registered Crop 1/"
		#path = "/Users/firetiti/Downloads/EM/Denoising/101/Registered Crop 2/"
		#path = "/Users/firetiti/Downloads/EM/Denoising/4373/Registered Crop 1024x1024/"
		dataset = Datasets.NRRN(path, normalizer=Normalizer.CenterReduce(MaxValue=255.0), nInputs=nInputs, train=True,
								datasetSize=opt.trainingSize, transformations=_InputTransform_(), Debug=opt.debug)
		if opt.debug:
			for i in range(dataset.__len__()):
				previm, im, nextim = dataset.__getitem__(i)
		print("Dataset created. Size = " + str(dataset.__len__()))
		
		TrainingDataLoader = DataLoader(dataset=dataset, num_workers=opt.nThreads, batch_size=opt.batchSize,
										shuffle=True)
		print("DataLoader created.")
		
		
		print(Colors.Colors.GREEN + '\n=====> Training' + Colors.Colors.RESET)
		nEpochs = int(opt.nIterations / (dataset.__len__() / opt.batchSize)) + 1
		print("Training configuration:")
		print(" - Minimum " + str(opt.nIterations) + " iterations/updates requested")
		print(" - Batch size = " + str(opt.batchSize))
		print(" ---> Number of epochs = " + str(nEpochs))
		print(" - Learning rate = " + str(opt.lr))
		print(" - Cuda = " + str(True if opt.cuda else False))
		print(" - Number of input image(s) = " + str(nInputs))
		
		for epoch in range(nEpochs):
			TrainSingleEpoch(model, optimizer, criterion, device, epoch, TrainingDataLoader)
			if scheduler is not None:
				scheduler.step()
		print("Training done.")
		
		year, month, day, hour, minutes = map(int, time.strftime("%Y %m %d %H %M").split())
		name = "Denoising_" + ModelType + "_nBlocks=" + str(opt.nBlocks) + "_nMaps=" + str(opt.nFeaturesMaps) + \
			"_Date=" + str(year) + "." + str(month) + "." + str(day) + "." + str(hour) + "h" + str(minutes) + ".pt"
		torch.save(model, name)
		print("Model saved as: " + name)
		
	
	
	
	print(Colors.Colors.GREEN + '\n=====> Densoising / Inference' + Colors.Colors.RESET)
	
	model.eval()
	
	print("Densoising / Inference configuration:")
	print(" - Batch size = " + str(opt.batchSize))
	print(" - Cuda = " + str(True if opt.cuda else False))
	print(" - Number of input image(s) = " + str(nInputs))
	
	path = opt.dataDir
	#path = "/Users/firetiti/Downloads/EM/Denoising/101/Registered Crop 1/"
	#path = "/Users/firetiti/Downloads/EM/Denoising/101/Registered Crop 2/"
	#path = "/Users/firetiti/Downloads/EM/Denoising/4373/Registered Crop 1024x1024/"
	if path[len(path)-1] == '/':
		path = path[0:len(path)-1]
		
	dataset = Datasets.NRRN(path, nInputs=nInputs, train=False,	Debug=opt.debug)
	
	normalizer = Normalizer.CenterReduce(MaxValue=255.0)
	
	denoiser = Denoiser.Denoiser(verbose=False)
	denoiser.Denoise(dataset, model, nInputs, 256, opt.nBlocks, opt.batchSize, device, normalizer, path + " - Denoised")
	#denoiser.Denoise(dataset, model, nInputs, 256, 0, opt.batchSize, device, normalizer, path + " - Denoised")
	
	print(Colors.Colors.GREEN + "\nAll Done." + Colors.Colors.RESET)



def Run(parameters=None):
	if parameters is not None:
		newArgs = [sys.argv[0]]
		keys = list(parameters.keys())
		for key in keys:
			newArgs.append('--' + key + '=' + str(parameters[key]))
		sys.argv = newArgs
	_Run_()












if __name__ == '__main__':
	parameters = {'dataDir': "/Users/firetiti/Downloads/EM/Denoising/4373-Pancreas-BC/Registered Crop 1024x1024/",
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
				'debug': False}
	#Run(parameters=parameters)
	Run()
