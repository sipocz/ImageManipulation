from  PIL import Image
from numpy import asarray
import pandas as pd
import scipy as sci
import matplotlib.pyplot  as plt

# load the image
image = Image.open('a.jpg')
# convert image to numpy array
data = asarray(image)
print(type(data))
print(data)
df=pd.DataFrame(data[:,:,0]+data[:,:,1]+data[:,:,2])
# summarize shape
print(data.shape)

# create Pillow image
image2 = Image.fromarray(data)
print(type(image2))

# summarize image details
print(image2.mode)
print(image2.size)
print (df)
plt.matshow(df)
plt.show()

