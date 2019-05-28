import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt

trdatafz='mnist/train-images-idx3-ubyte.gz'
trlabelfz='mnist/train-labels-idx1-ubyte.gz'
tstdatafz='mnist/t10k-images-idx3-ubyte.gz'
tstlabelfz='mnist/t10k-labels-idx1-ubyte.gz'

def readim(fz):
    with gzip.open(fz, 'rb') as f:
        header=f.read(16)
        mn,num,nrow,ncol=struct.unpack('>4i',header)
        assert mn == 2051
        im=np.empty((num,nrow,ncol))
        npixel=nrow*ncol
        for i in range(num):
            buf=struct.unpack('>%dB' % npixel, f.read(npixel))
            im[i,:,:]=np.asarray(buf).reshape((nrow,ncol))
    return im

def readlabel(fz):
    with gzip.open(fz,'rb') as f:
        header=f.read(8)
        mn,num=struct.unpack('>2i', header)
        assert mn == 2049
        label = np.array(struct.unpack('>%dB' % num, f.read()),dtype=int)
    return label

if __name__ == '__main__':
    """
    trlabel=readlabel(trlabelfz)
    trdata=readim(trdatafz)
    """
    tstlabel=readlabel(tstlabelfz)
    tstdata=readim(tstdatafz)
    plt.figure()
    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(tstdata[i,:,:],cmap='gray')
        plt.title(tstlabel[i])
    #plt.savefig('mnist.eps')
    plt.show()
