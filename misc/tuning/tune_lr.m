clear
close all;

lr = [0.000010
0.000012
0.000013
0.000015
0.000017
0.000020
0.000023
0.000027
0.000031
0.000035
0.000040
0.000047
0.000054
0.000062
0.000071
0.000081
0.000094
0.000108
0.000124
0.000142
0.000164
0.000188
0.000216
0.000249
0.000286
0.000329
0.000379
0.000435
0.000501
0.000576
0.000662
0.000761
0.000876
0.001007
0.001158
0.001332
0.001532
0.001761
0.002025
0.002329
0.002679
0.003080
0.003542
0.004074
0.004685
0.005388
0.006196
0.007125
0.008194
0.009423
0.010837
0.012462
0.014331
0.016481
0.018953
0.021796
0.025066
0.028826
0.033149
0.038122
0.043840
0.050416
0.057978
0.066675
0.076676
0.088178
0.101405
0.116615
0.134108
0.154224
0.177357
0.203961
0.234555
0.269738
0.310199
0.356729
0.410238
0.471774
0.542540
0.623921
0.717509
0.825135
0.948905
1.091241
1.254927
1.443166
1.659641
1.908588
2.194876
2.524107
2.902723
3.338132
3.838852
4.414679
5.076881
5.838413
6.714175
7.721302
8.879497
10.211421];

loss = [12.118391
12.126423
12.101704
12.110937
12.099863
12.107471
12.108615
12.100868
12.105974
12.112607
12.107846
12.092377
12.120878
12.116482
12.122443
12.101524
12.120376
12.108444
12.130797
12.106287
12.091908
12.109046
12.120393
12.118368
12.122268
12.111627
12.103424
12.114486
12.114756
12.101926
12.117523
12.122892
12.112489
12.097409
12.102818
12.112334
12.111358
12.104523
12.109972
12.122058
12.099743
12.093750
12.108168
12.097073
12.082552
12.106123
12.067334
12.098247
12.092596
12.069508
12.091114
12.038979
12.104502
12.027763
12.022923
11.964986
12.089109
12.002943
12.047002
11.999248
12.013765
11.992388
11.994008
11.752586
11.741868
11.794216
11.967196
11.666634
11.758560
11.617725
11.744757
11.581100
11.750719
11.916762
11.559115
11.723582
11.489686
11.469151
11.965362
11.624237
12.014537
11.662418
11.676790
12.079234
12.095613
12.226258
12.356383
12.456889
12.382455
12.655602
12.642792
12.701313
12.878495
12.924626
12.925916
12.694982
12.494555
12.119529
11.976392
11.884874];

sma = 1;
derivatives = (loss(1+sma:end) - loss(1:end-sma))/sma;
derivatives = filter(ones(1,5)/5,1,derivatives);
figure();
semilogx(lr, loss);
figure();
semilogx(lr(2:end), derivatives)
