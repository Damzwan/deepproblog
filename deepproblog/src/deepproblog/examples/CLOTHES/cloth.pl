top(0).
top(2).
top(6).

bot(1).
bot(3).
bot(8).

shoe(5).
shoe(7).
shoe(9).

clothesGroup(A, B, C) :- clothes(A, C1), clothes(B, C2), clothes(C, C3),
    (top(C1), bot(C2), shoe(C3); top(C1), bot(C2), shoe(C3); top(C2), bot(C1), shoe(C3);
    top(C2), bot(C3), shoe(C1); top(C3), bot(C1), shoe(C2); top(C3), bot(C2), shoe(C1)).

nn(cloth_mnist_net,[X],Y,[0,1,2,3,5,6,7,8,9]) :: clothes(X, Y).