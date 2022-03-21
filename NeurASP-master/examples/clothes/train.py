import sys
sys.path.append('../../')
import time

import torch

from datagen import dataList, obsList, test_loader
from network import Net
from neurasp import NeurASP

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = r'''
clothesImage(c1). 
clothesImage(c2).
clothesImage(c3).
top(0).
bot(1).
shoe(5).

clothesGroup(A, B, C) :- top(A), bot(B), shoe(C); top(A), bot(C), shoe(B); top(B), bot(A), shoe(C);
                         top(B), bot(C), shoe(A); top(C), bot(A), shoe(B); top(C), bot(B), shoe(A).
                         
finalQuery(C1, C2, C3) :- clothes(0, C1, A), clothes(0, C2, B), clothes(0, C3, C), 
                       A != B, B != C, A != C, 
                       clothesGroup(A, B, C).
                       
finalQuery(c1, c2, c3).
nn(clothes(1, X), [0, 1, 5]) :- clothesImage(X).
'''


########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net()
nnMapping = {'clothes': m}
optimizers = {'clothes': torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

saveModelPath = 'data/model.pt'

for i in range(1):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle='data/stableModels.pickle')
    time2 = time.time()
    acc, _ = NeurASPobj.testNN('clothes', test_loader)
    print('Test Acc: {:0.2f}%'.format(acc))
    print('Storing the trained model into {}'.format(saveModelPath))
    torch.save(m.state_dict(), saveModelPath)
    print('--- train time: %s seconds ---' % (time2 - time1))
    print('--- test time: %s seconds ---' % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )