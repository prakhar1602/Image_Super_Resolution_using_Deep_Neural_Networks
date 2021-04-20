from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation

class srcnn(Model):
	def __init__(self, img_size, classes):
		super(srcnn, self).__init__()
		self.img_size = img_size
		self.classes = classes
		
		self.conv1 = Conv2D(64,9,padding='same', input_shape = (self.img_size, self.img_size, self.classes), activation='relu')
		self.conv2 = Conv2D(32,1,padding='same', activation='relu')
		self.conv3 = Conv2D(self.classes, 5, padding='same')
		
	def call(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		return x
