from torch import nn

class srcnnModel(nn.Module):
	def __init__(self, num_channels=1, kernel_size_l1 = 9, kernel_size_l2=5, kernel_size_l3=5):
		super(srcnnModel, self).__init__()
		self.layer1 = nn.Conv2d(num_channels, 64, kernel_size=kernel_size_l1, padding = kernel_size_l1//2)
		self.layer2 = nn.Conv2d(64, 32, kernel_size=kernel_size_l2, padding = kernel_size_l2//2)
		self.layer3 = nn.Conv2d(32, num_channels, kernel_size=kernel_size_l3, padding = kernel_size_l3//2)
		self.relu = nn.Relu(inplace=True)

	def forward(self, x):
		x = self.relu(self.layer1(x))
		x = self.relu(self.layer2(x))
		x = self.layer3(x)
		return x
	
