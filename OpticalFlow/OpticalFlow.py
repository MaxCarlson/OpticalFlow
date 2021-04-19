from PIL import Image
import numpy as np

f1a = Image.open('frame1_a.png')
f2a = Image.open('frame2_a.png')

f1b = Image.open('frame1_b.png')
f2b = Image.open('frame2_b.png')

f1a = np.asarray(f1a)
f2a = np.asarray(f2a)

f1b = np.asarray(f1b)
f2b = np.asarray(f2b)


class OpticalFlow:
    dogx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    dogy = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])

    def __init__(self, images, dims=images[0].shape):
        self.images     = images
        self.dtemporal  = images[1] - images[0] 

    def applyFilter(self, filter, padding):
        im = self.images[0]
        im = np.pad(im, 1)
        sz = im.shape
        for i in range(0, sz[0] - padding * 2):
            for j in range(0, sz[1] - padding * 2):
                z = np.multiply(filter, im[i:i+fz, j:j+fz])
                out[i][j] = np.sum(z)


    def run(self):
        # Calculate DoGx and DoGy for each pixel
        Ix = self.applyFilter(dogx, 1)
        Iy = self.applyFilter(dogy, 1)





of = OpticalFlow([f1a, f2a])

#of = OpticalFlow([f1b, f2b])
of.run()