from DataHandling import *


## The modified IoU Loss
def re_Dice_Loss(inputs, targets, cuda=False, balance=1.1):
	n, c, h, w = inputs.size()
	smooth=1
	inputs = torch.sigmoid(inputs)  # F.sigmoid(inputs)

	input_flat=inputs.view(-1)
	target_flat=targets.view(-1)

	intersecion=input_flat*target_flat
	unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
	loss=unionsection/(2*intersecion.sum()+smooth)
	loss=loss.sum()

	return loss

## Weight Binary Cross Entropy Loss
def _weighted_cross_entropy_loss(preds, edges, device, weight = 10):
	mask = (edges == 1.0).float()
	b, c, h, w = edges.shape
	num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
	num_neg = c * h * w - num_pos                     # Shape: [b,].
	weight = torch.zeros_like(edges)
	nx1 = num_neg / (num_pos + num_neg)
	nx2 = num_pos / (num_pos + num_neg)
	weight = torch.cat([torch.where(i == 1.0, j, k) for i, j, k in zip(edges, nx1, nx2)], dim = 0).unsqueeze(1)
	# Calculate loss
	losses = F.binary_cross_entropy_with_logits(preds.float(),
												edges.float(),
												weight=weight,
												reduction='none')
	loss = torch.sum(losses) / b
	return loss

### The following function is taken from the original PyTorch DCGAN Tutorial. I have made minor changes to the code to save the checkpoint files for both the Generator and the Discriminator. We only use the Generator to visualize the results so the Discriminator's checkpoint is not made available in the zip file. 
def training(num_epochs, eno, modelG, modelD, dataloader, cuda, criterion, criterion2, optimizerG, optimizerD, scheduler = None, schedulerD = None, f_name = 'model.pt'):
	modelG.train()
	modelD.train()
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0
	print("Starting Training Loop...")
	for epoch in range(num_epochs):
		for i, data in enumerate(dataloader):
			modelD.zero_grad()
			real_cpu = data[0].to(cuda)
			real_contour = data[2].to(cuda)
			b_size = real_cpu.size(0)
			label = torch.full((b_size,), 1., dtype=torch.float, device=cuda)
			output = modelD(torch.cat([real_cpu, real_contour], dim = 1)).view(-1)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()
			fake = modelG(real_cpu)
			label.fill_(0.)
			output = modelD(torch.cat([real_cpu, fake.detach()], dim = 1)).view(-1)
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()


			modelG.zero_grad()
			label.fill_(1.)
			output = modelD(torch.cat([real_cpu, fake], dim = 1)).view(-1)
			errG = criterion(output, label)
			loss = _weighted_cross_entropy_loss(fake, real_contour, cuda)
			loss_small = re_Dice_Loss(fake, real_contour)
			errG += (0.001 * loss + loss_small)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
				% (epoch, num_epochs, i, len(dataloader),
				errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


			G_losses.append(errG.item())
			D_losses.append(errD.item())

			iters += 1

		if scheduler is not None:
			scheduler.step()
		if schedulerD is not None:
			schedulerD.step()
		if epoch % 5 != 0 and epoch != 0:
			continue
		torch.save({
			'epoch': epoch + eno,
			'model_state_dict': modelG.module.state_dict(),
			'optimizer_state_dict': optimizerG.state_dict(),
			'loss': errG.item()
		}, f = "checkpoints/GeneratorXX" + str(epoch + eno) + '.pt')
		print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
			% (epoch, num_epochs, i, len(dataloader),
			errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), flush = True)
		torch.save({
			'epoch': epoch + eno,
			'model_state_dict': modelD.module.state_dict(),
			'optimizer_state_dict': optimizerD.state_dict(),
			'loss': errD.item()
		}, f = "checkpoints/DiscriminatorX" + str(epoch + eno) + '.pt')
	print("Training Finished")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size = 3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, padding=kernel_size//2, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size= kernel_size, padding=kernel_size//2, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

## The U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		self.model = UNet(3, 1)

	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):
	"""docstring for Discriminator."""

	def __init__(self):
		super(Discriminator, self).__init__()
		self.model = torchvision.models.resnet18(pretrained = False)
		self.model.conv1 = nn.Conv2d(4, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
		self.model.fc = nn.Linear(self.model.fc.in_features, 1)
		for param in self.model.parameters():
			param.requires_grad = True

	def forward(self, x):
		return self.model(x)


def main(train = False, lr = 0.001, epochs = 30, t = 25, f_name = 'checkpoints/model.pt', device_list = None, device = 0, batch = 0, sched = 1):
	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
	modelG = Generator()
	modelD = Discriminator()
	init_weights(modelG)
	if device_list is not None:
		modelG = nn.DataParallel(modelG, device_ids = device_list)
		modelD = nn.DataParallel(modelD, device_ids = device_list)
	else:
		print(device, flush = True)
	modelG.to(cuda)
	modelD.to(cuda)
	print(cuda, flush = True)
	transforms_ = transforms.ToTensor()
	train_set = AugmentedDataset(train_type = 'train')
	print(len(train_set), flush = True)
	train_dataloader = DataLoader(train_set, batch_size = batch, shuffle = True, num_workers = 8)
	optimizerG = torch.optim.Adam(modelG.parameters(), lr = lr)
	optimizerD = torch.optim.Adam(modelD.parameters(), lr = 0.002)
	if sched:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[5, 10, 15, 20], gamma=0.1)
		schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[5, 10, 15, 20], gamma=0.1)
	else:
		scheduler = None
		schedulerD = None
	criterion2 = nn.L1Loss()
	criterion = nn.BCEWithLogitsLoss()
	epoch = 0
	try:
		checkpoint = torch.load('Generator.pt')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print('Checkpoint Available!', flush = True)
		start = time.time()
	except:
		print('No Pre-Training', flush = True)
		start = time.time()

	if train:
			model = training(abs(epochs - epoch), abs(epoch - t), modelG, modelD, train_dataloader, cuda, criterion, criterion2, optimizerG, optimizerD, scheduler, schedulerD, f_name = f_name)
			end = time.time()
			print('Time taken for %d with a batch_size of %d is %.2f hours.' %(epochs, batch, (end - start) / (3600)), flush = True)

### The below lines of code should not be commented for training (uncomment those lines for training) and need to be commented when running the .ipynb file to visualize the results. ###

'''lr = float(sys.argv[1])
n = int(sys.argv[5])
b = int(sys.argv[6])
if n > 1:
	main(train = True, lr = lr, epochs = int(sys.argv[2]), t = int(sys.argv[3]), f_name = 'checkpoints/' + str(sys.argv[4]), device_list = [i for i in range(n)], batch = n * b, sched = int(sys.argv[-1]))
else:
	main(train = True, lr = lr, epochs = int(sys.argv[2]), t = int(sys.argv[3]), f_name = 'checkpoints/' + str(sys.argv[4]), device = n, batch = b, sched = int(sys.argv[-1]))'''
