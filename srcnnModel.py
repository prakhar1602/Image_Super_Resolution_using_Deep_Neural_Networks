from torch import nn

class srcnnModel(nn.Module):
	def __init__(self, num_channels=1):
		super(srcnnModel, self).__init__()
		self.layer1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding = 9//2)
		self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding = 5//2)
		self.layer3 = nn.Conv2d(32, num_channels, kernel_size=5, padding = 5//2)
		self.relu = nn.Relu(inplace=True)

	def forward(self, x):
		x = self.relu(self.layer1(x))
		x = self.relu(self.layer2(x))
		x = self.layer3(x)
		return x
	