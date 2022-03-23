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
clothesImage(c1; c2; c3). 
top(0; 2; 6).
bot(1; 3; 8).
shoe(5; 7; 9).

% "winand dit is super lelijk" -> https://stackoverflow.com/questions/66880574/disjunction-in-the-body-of-a-rule-in-clingo
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(A), bot(B), shoe(C).
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(A), bot(C), shoe(B).
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(B), bot(A), shoe(C).
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(B), bot(C), shoe(A).
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(C), bot(A), shoe(B).
clothesGroup(c1, c2, c3, 1) :- clothes(0, CX, A), clothes(0, CY, B), clothes(0, CZ, C), top(C), bot(B), shoe(A).

clothesGroup(c1, c2, c3, 0) :- not clothesGroup(c1, c2, c3, 1).
                       
nn(clothes(1, X), [0, 1, 5]) :- clothesImage(X).
'''


########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net(3) # write amount of possible clothes classes here
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