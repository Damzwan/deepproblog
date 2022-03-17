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
clothes(Y) :- top(Y); bot(Y); shoe(Y).

clothesGroup(A, B, C) :- top(A), bot(B), shoe(C); top(A), bot(C), shoe(B); top(B), bot(A), shoe(C);
                         top(B), bot(C), shoe(A); top(C), bot(A), shoe(B); top(C), bot(B), shoe(A).
nn(clothes(1, X), [0, 1, 2, 3, 5, 6, 7, 8, 9]) :- clothesImage(X).
'''

# findall(X, is_top(X), ListT), length(ListT, 1),
#                          findall(X, is_bot(X), ListB), length(ListB, 1),
#                          findall(X, is_shoe(X), ListS), length(ListS, 1).

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