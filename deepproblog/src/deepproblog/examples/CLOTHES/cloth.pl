nn(cloth_mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: clothes(X, Y).

clothesGroup(A, B, C) :- clothes(A, C1),clothes(B, C2), clothes(C, C3), top(A), bot(B), shoe(C); top(A), bot(C), shoe(B); top(B), bot(A), shoe(C);
                         top(B), bot(C), shoe(A); top(C), bot(A), shoe(B); top(C), bot(B), shoe(A).