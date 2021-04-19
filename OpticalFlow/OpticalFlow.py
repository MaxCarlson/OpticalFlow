from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, images):
        self.images     = images
        self.images[0] = np.asarray(self.images[0], dtype=int)
        self.images[1] = np.asarray(self.images[1], dtype=int)

        self.images[0]  = np.pad(self.images[0], 1)
        self.dtemporal  = np.pad(images[1], 1) - images[0] 

    def applyFilter(self, filter, padding):
        im = self.images[0]
        sz = im.shape
        fz = filter.shape[0]
        out = np.zeros(self.images[1].shape)

        for i in range(0, sz[0] - padding * 2):
            for j in range(0, sz[1] - padding * 2):
                z = np.multiply(filter, im[i:i+fz, j:j+fz])
                out[i][j] = np.sum(z)
        return out


    def run(self):
        # Calculate DoGx and DoGy for each pixel
        Ix = self.applyFilter(self.dogx, 1)
        Iy = self.applyFilter(self.dogy, 1)
        Ix = np.pad(Ix, 1)
        Iy = np.pad(Iy, 1)

        V = np.zeros((self.images[1].shape + (2,)))

        for i in range(self.images[1].shape[0]):
            for j in range(self.images[1].shape[1]):
                # Build the DoGx and DoGy matricies for each pixel
                xs = np.reshape(Ix[i:i+3, j:j+3], (9))
                ys = np.reshape(Iy[i:i+3, j:j+3], (9))
                a = np.transpose(np.stack([xs, ys]))

                # as well as the temporal b term
                b = -np.reshape(self.dtemporal[i:i+3, j:j+3], (9))

                ata = np.matmul(np.transpose(a), a)
                atb = np.matmul(np.transpose(a), b)

                #r = np.matmul(np.linalg.inv(ata), atb)
                try:
                    r = np.matmul(np.linalg.inv(ata), atb)
                except:
                    r = np.matmul(np.linalg.pinv(ata), atb)

                V[i, j] = r

        Vx = V[:,:,0]
        Vy = V[:,:,1]

        V2 = [np.sqrt(np.square(v[0]) + np.square(v[1])) 
              for v in np.reshape(V, (V.shape[0] * V.shape[1], 2))]
        V2 = np.reshape(V2, self.images[1].shape)

        plt.imshow(V2)
        plt.show()

        vxImg = Image.fromarray(Vx)
        vyImg = Image.fromarray(Vy)
        v2Img = Image.fromarray(V2)
        
        vxImg.show()
        vyImg.show()
        v2Img.show()
        a=5
        








of = OpticalFlow([f1a, f2a])

#of = OpticalFlow([f1b, f2b])
of.run()