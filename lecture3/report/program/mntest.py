import numpy as np
import matplotlib.pyplot as plt
import mnread

def train(label,data):
    data = np.reshape(data,[data.shape[0],-1])
    lset=set(label)
    model=np.empty((len(lset),data.shape[1]),dtype=float)
    for x in lset:
        model[x,:] = np.mean(data[np.where(label==x),:],axis=1)
    return model

def classify(data,model):
    data = np.reshape(data,[data.shape[0],-1])
    label = np.empty(data.shape[0],dtype=int)
    for i in range(data.shape[0]):
        label[i] = np.argmin(np.sum((model-data[i,:])**2,axis=1))
    return label

trlabel = mnread.readlabel(mnread.trlabelfz)
trdata = mnread.readim(mnread.trdatafz)
tstlabel = mnread.readlabel(mnread.tstlabelfz)
tstdata = mnread.readim(mnread.tstdatafz)

model = train(trlabel, trdata)
estlabel = classify(tstdata, model)
print('accuracy: %g' % (float(sum(estlabel==tstlabel)) / len(tstlabel)))

plt.figure()
plt.suptitle('goods')
goods = np.random.permutation(np.where(estlabel==tstlabel)[-1])[range(50)]
for i,good in enumerate(goods):
    plt.subplot(5,10,i+1)
    plt.axis('off')
    plt.imshow(tstdata[good,:,:],cmap='gray')
    plt.title(estlabel[good])
plt.savefig('good.eps')
plt.figure()
plt.suptitle('bads')
bads = np.random.permutation(np.where(~(estlabel==tstlabel))[-1])[range(50)]
for i,bad in enumerate(bads):
    plt.subplot(5,10,i+1)
    plt.axis('off')
    plt.imshow(tstdata[bad,:,:],cmap='gray')
    plt.title('%s(%s)' % (estlabel[bad], tstlabel[bad]))
plt.savefig('bad.eps')
plt.show()
