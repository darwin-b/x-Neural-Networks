# x-Neural-Networks

#### Note  
    [Runtime - Change runtime type - Hardware accelerator - GPU]
      - If experiencing memory issues on google colab switch to Kaggle notebook and use gpu as accelerator.
#### FILE
>   nn.py
#### DESCRIPTION
    This is the start of an exceedingly simple / lite / reduced functionality PyTorch style xNN library written in Python and it's example  use for MNIST image classification
    This code does not use PyTorch, TensorFlow or any other xNN library
    
       NN specification:
       ----------------------------   -------
       Data loader                    Output
       ----------------------------   -------
       Data                           1x28x28
       Division by 255.0              1x28x28

       ----------------------------   -------
       Network                        Output
       ----------------------------   -------
       Vectorization                  1x784
       Vector matrix multiplication   1x100
       Vector vector addition         1x100
       ReLU                           1x100
       Vector matrix multiplication   1x100
       Vector vector addition         1x100
       ReLU                           1x100
       Vector matrix multiplication   1x10
       Vector vector addition         1x10

       ----------------------------   -------
       Error                          Output
       ----------------------------   -------
       Softmax                        1x10
       Cross entropy                  1

####   Output generated during training showing the per epoch statistics
    
       Epoch   0 Time     32.0 lr = 0.000100 avg loss = 1.114983 accuracy = 86.70
       Epoch   1 Time     32.1 lr = 0.003400 avg loss = 0.226087 accuracy = 95.40
       Epoch   2 Time     32.0 lr = 0.006700 avg loss = 0.130239 accuracy = 96.12
       Epoch   3 Time     32.0 lr = 0.010000 avg loss = 0.109192 accuracy = 93.86
       Epoch   4 Time     31.9 lr = 0.009699 avg loss = 0.073910 accuracy = 96.36
       Epoch   5 Time     31.9 lr = 0.008831 avg loss = 0.052902 accuracy = 97.40
       Epoch   6 Time     31.9 lr = 0.007502 avg loss = 0.036914 accuracy = 97.87
       Epoch   7 Time     31.9 lr = 0.005872 avg loss = 0.022554 accuracy = 97.85
       Epoch   8 Time     31.9 lr = 0.004138 avg loss = 0.013180 accuracy = 98.18
       Epoch   9 Time     31.9 lr = 0.002507 avg loss = 0.008136 accuracy = 98.30
       Epoch  10 Time     32.1 lr = 0.001179 avg loss = 0.005525 accuracy = 98.31
       Epoch  11 Time     32.0 lr = 0.000311 avg loss = 0.004485 accuracy = 98.29
       Epoch  12 Time     31.9 lr = 0.000010 avg loss = 0.004097 accuracy = 98.28



#### FILES
>   eff.py 

>   IR-Net_NetworksPaper.pdf

#### DESCRIPTION
    A PyTorch implementation of the network described in section 3 of IR-Net_NetworksPaper.doc/pdf. 
    Designed adn trained the network in table 1 of the paper with the standard inverted residual building block in fig 2a .

    Trained in Google Colaboratory using a GPU instance.
    (Runtime - Change runtime type - Hardware accelerator - GPU)

#   Output generated during training 
    Showing the per epoch statistics (for networks trained using standard network block as described in paper)

    Using 1 GPU(s)
      Epoch   0 Time    372.6 lr = 0.002000 avg loss = 0.016492 accuracy = 11.42
      Epoch   1 Time    371.4 lr = 0.041600 avg loss = 0.014386 accuracy = 21.76
      Epoch   2 Time    371.6 lr = 0.081200 avg loss = 0.012312 accuracy = 27.72
      Epoch   3 Time    372.1 lr = 0.120800 avg loss = 0.010926 accuracy = 33.18
      Epoch   4 Time    369.9 lr = 0.160400 avg loss = 0.010022 accuracy = 36.96
      Epoch   5 Time    368.1 lr = 0.200000 avg loss = 0.009387 accuracy = 42.26
      Epoch   6 Time    367.8 lr = 0.199795 avg loss = 0.008718 accuracy = 44.80
      Epoch   7 Time    367.2 lr = 0.199180 avg loss = 0.008228 accuracy = 42.88
      Epoch   8 Time    369.6 lr = 0.198158 avg loss = 0.007870 accuracy = 49.44
      Epoch   9 Time    371.6 lr = 0.196733 avg loss = 0.007578 accuracy = 51.80
      Epoch  10 Time    370.8 lr = 0.194911 avg loss = 0.007316 accuracy = 49.36
      Epoch  11 Time    370.4 lr = 0.192699 avg loss = 0.007089 accuracy = 54.32
      Epoch  12 Time    368.3 lr = 0.190107 avg loss = 0.006901 accuracy = 55.62
      Epoch  13 Time    366.7 lr = 0.187145 avg loss = 0.006747 accuracy = 54.24
      Epoch  14 Time    366.1 lr = 0.183825 avg loss = 0.006571 accuracy = 55.92
      Epoch  15 Time    365.7 lr = 0.180161 avg loss = 0.006419 accuracy = 58.14
      Epoch  16 Time    366.4 lr = 0.176168 avg loss = 0.006316 accuracy = 60.32
      Epoch  17 Time    366.9 lr = 0.171863 avg loss = 0.006190 accuracy = 60.66
      Epoch  18 Time    365.9 lr = 0.167263 avg loss = 0.006040 accuracy = 59.88
      Epoch  19 Time    365.2 lr = 0.162387 avg loss = 0.005930 accuracy = 62.44
      Epoch  20 Time    364.8 lr = 0.157254 avg loss = 0.005848 accuracy = 56.64
      Epoch  21 Time    363.8 lr = 0.151887 avg loss = 0.005756 accuracy = 58.38
      Epoch  22 Time    363.9 lr = 0.146308 avg loss = 0.005635 accuracy = 62.72
      Epoch  23 Time    364.2 lr = 0.140538 avg loss = 0.005545 accuracy = 63.94
      Epoch  24 Time    364.0 lr = 0.134602 avg loss = 0.005430 accuracy = 61.74
      Epoch  25 Time    364.0 lr = 0.128524 avg loss = 0.005346 accuracy = 65.00
      Epoch  26 Time    364.6 lr = 0.122330 avg loss = 0.005256 accuracy = 63.78
      Epoch  27 Time    365.5 lr = 0.116044 avg loss = 0.005156 accuracy = 63.80
      Epoch  28 Time    364.5 lr = 0.109693 avg loss = 0.005053 accuracy = 64.36
      Epoch  29 Time    365.1 lr = 0.103302 avg loss = 0.004952 accuracy = 64.64
      Epoch  30 Time    366.0 lr = 0.096898 avg loss = 0.004860 accuracy = 65.34
      Epoch  31 Time    367.2 lr = 0.090507 avg loss = 0.004746 accuracy = 66.04
      Epoch  32 Time    366.8 lr = 0.084156 avg loss = 0.004648 accuracy = 67.04
      Epoch  33 Time    367.2 lr = 0.077870 avg loss = 0.004560 accuracy = 64.96
      Epoch  34 Time    366.6 lr = 0.071676 avg loss = 0.004465 accuracy = 65.20
      Epoch  35 Time    368.5 lr = 0.065598 avg loss = 0.004338 accuracy = 67.06
      Epoch  36 Time    370.2 lr = 0.059662 avg loss = 0.004252 accuracy = 67.80
      Epoch  37 Time    370.9 lr = 0.053892 avg loss = 0.004145 accuracy = 65.28
      Epoch  38 Time    368.8 lr = 0.048313 avg loss = 0.003987 accuracy = 67.50
      Epoch  39 Time    366.7 lr = 0.042946 avg loss = 0.003907 accuracy = 69.88
      Epoch  40 Time    367.1 lr = 0.037813 avg loss = 0.003780 accuracy = 70.14
      Epoch  41 Time    366.7 lr = 0.032937 avg loss = 0.003665 accuracy = 70.34
      Epoch  42 Time    366.6 lr = 0.028337 avg loss = 0.003567 accuracy = 70.52
      Epoch  43 Time    366.7 lr = 0.024032 avg loss = 0.003479 accuracy = 70.58
      Epoch  44 Time    368.7 lr = 0.020039 avg loss = 0.003360 accuracy = 70.56
      Epoch  45 Time    369.1 lr = 0.016375 avg loss = 0.003256 accuracy = 70.90
      Epoch  46 Time    369.8 lr = 0.013055 avg loss = 0.003167 accuracy = 71.80
      Epoch  47 Time    369.3 lr = 0.010093 avg loss = 0.003093 accuracy = 71.20
      Epoch  48 Time    367.7 lr = 0.007501 avg loss = 0.003029 accuracy = 72.30
      Epoch  49 Time    367.1 lr = 0.005289 avg loss = 0.002953 accuracy = 72.64
      Epoch  50 Time    367.3 lr = 0.003467 avg loss = 0.002935 accuracy = 72.34
      Epoch  51 Time    366.7 lr = 0.002042 avg loss = 0.002890 accuracy = 72.60
      Epoch  52 Time    365.8 lr = 0.001020 avg loss = 0.002871 accuracy = 72.40
      Epoch  53 Time    365.8 lr = 0.000405 avg loss = 0.002857 accuracy = 72.54
      Epoch  54 Time    367.2 lr = 0.000200 avg loss = 0.002847 accuracy = 72.36


#### FILES
>   eff_se.py

>   IR-Net_NetworksPaper.pdf

#### DESCRIPTION
    A PyTorch implementation of the network described in section 3 of IR-Net_NetworksPaper.doc/pdf.
    Designed and trained the network in table 1 of the paper with the SE enhanced building block in fig 2b .
    
    Trained in Google Colaboratory using a GPU instance.
    (Runtime - Change runtime type - Hardware accelerator - GPU)

#   Output generated during training 
    Showing the per epoch statistics (for networks trained using standard network block as described in paper)

    Using 1 GPU(s)
      Epoch   0 Time     32.9 lr = 0.002000 avg loss = 0.017915
      Epoch   0 Time     31.4 lr = 0.002000 avg loss = 0.017757
      Epoch   0 Time     31.6 lr = 0.002000 avg loss = 0.017608
      Epoch   0 Time     31.8 lr = 0.002000 avg loss = 0.017473
      Epoch   0 Time     31.9 lr = 0.002000 avg loss = 0.017352
      Epoch   0 Time     31.9 lr = 0.002000 avg loss = 0.017237
      Epoch   0 Time     31.8 lr = 0.002000 avg loss = 0.017109
      Epoch   0 Time     31.9 lr = 0.002000 avg loss = 0.016977
      Epoch   0 Time     31.8 lr = 0.002000 avg loss = 0.016855
      Epoch   0 Time     31.8 lr = 0.002000 avg loss = 0.016725
      Epoch   0 Time     31.6 lr = 0.002000 avg loss = 0.016610
      Epoch   0 Time     31.9 lr = 0.002000 avg loss = 0.016500
      Epoch   0 Time    407.9 lr = 0.002000 avg loss = 0.016433 accuracy = 12.46
      Epoch   1 Time     58.3 lr = 0.041600 avg loss = 0.016576
      Epoch   1 Time     31.5 lr = 0.041600 avg loss = 0.016099
      Epoch   1 Time     31.2 lr = 0.041600 avg loss = 0.015811
      Epoch   1 Time     31.3 lr = 0.041600 avg loss = 0.015574
      Epoch   1 Time     31.5 lr = 0.041600 avg loss = 0.015385
      Epoch   1 Time     31.6 lr = 0.041600 avg loss = 0.015225
      Epoch   1 Time     31.4 lr = 0.041600 avg loss = 0.015073
      Epoch   1 Time     31.3 lr = 0.041600 avg loss = 0.014929
      Epoch   1 Time     31.4 lr = 0.041600 avg loss = 0.014805
      Epoch   1 Time     31.4 lr = 0.041600 avg loss = 0.014674
      Epoch   1 Time     31.3 lr = 0.041600 avg loss = 0.014568
      Epoch   1 Time     31.4 lr = 0.041600 avg loss = 0.014447
      Epoch   1 Time    403.6 lr = 0.041600 avg loss = 0.014381 accuracy = 20.40
      Epoch   2 Time     58.4 lr = 0.081200 avg loss = 0.013416
      Epoch   2 Time     31.3 lr = 0.081200 avg loss = 0.013294
      Epoch   2 Time     31.5 lr = 0.081200 avg loss = 0.013200
      Epoch   2 Time     31.5 lr = 0.081200 avg loss = 0.013065
      Epoch   2 Time     31.5 lr = 0.081200 avg loss = 0.012954
      Epoch   2 Time     31.6 lr = 0.081200 avg loss = 0.012845
      Epoch   2 Time     31.3 lr = 0.081200 avg loss = 0.012753
      Epoch   2 Time     31.5 lr = 0.081200 avg loss = 0.012645
      Epoch   2 Time     31.3 lr = 0.081200 avg loss = 0.012552
      Epoch   2 Time     31.2 lr = 0.081200 avg loss = 0.012466
      Epoch   2 Time     31.2 lr = 0.081200 avg loss = 0.012375
      Epoch   2 Time     31.2 lr = 0.081200 avg loss = 0.012282
      Epoch   2 Time    403.1 lr = 0.081200 avg loss = 0.012224 accuracy = 29.78
      Epoch   3 Time     58.0 lr = 0.120800 avg loss = 0.011494
      Epoch   3 Time     31.3 lr = 0.120800 avg loss = 0.011445
      Epoch   3 Time     31.4 lr = 0.120800 avg loss = 0.011358
      Epoch   3 Time     31.3 lr = 0.120800 avg loss = 0.011256
      Epoch   3 Time     31.2 lr = 0.120800 avg loss = 0.011170
      Epoch   3 Time     31.2 lr = 0.120800 avg loss = 0.011087
      Epoch   3 Time     31.4 lr = 0.120800 avg loss = 0.011008
      Epoch   3 Time     31.3 lr = 0.120800 avg loss = 0.010935
      Epoch   3 Time     31.3 lr = 0.120800 avg loss = 0.010869
      Epoch   3 Time     31.1 lr = 0.120800 avg loss = 0.010815
      Epoch   3 Time     31.3 lr = 0.120800 avg loss = 0.010769
      Epoch   3 Time     31.2 lr = 0.120800 avg loss = 0.010709
      Epoch   3 Time    401.8 lr = 0.120800 avg loss = 0.010670 accuracy = 31.24
      Epoch   4 Time     57.8 lr = 0.160400 avg loss = 0.010227
      Epoch   4 Time     31.1 lr = 0.160400 avg loss = 0.010178
      Epoch   4 Time     31.5 lr = 0.160400 avg loss = 0.010116
      Epoch   4 Time     31.5 lr = 0.160400 avg loss = 0.010101
      Epoch   4 Time     31.6 lr = 0.160400 avg loss = 0.010061
      Epoch   4 Time     31.5 lr = 0.160400 avg loss = 0.010011
      Epoch   4 Time     31.4 lr = 0.160400 avg loss = 0.009975
      Epoch   4 Time     31.4 lr = 0.160400 avg loss = 0.009928
      Epoch   4 Time     31.4 lr = 0.160400 avg loss = 0.009883
      Epoch   4 Time     31.4 lr = 0.160400 avg loss = 0.009831
      Epoch   4 Time     31.5 lr = 0.160400 avg loss = 0.009790
      Epoch   4 Time     31.5 lr = 0.160400 avg loss = 0.009761
      Epoch   4 Time    403.9 lr = 0.160400 avg loss = 0.009735 accuracy = 40.88
      Epoch   5 Time     58.8 lr = 0.200000 avg loss = 0.009334
      Epoch   5 Time     31.6 lr = 0.200000 avg loss = 0.009360
      Epoch   5 Time     31.5 lr = 0.200000 avg loss = 0.009352
      Epoch   5 Time     31.6 lr = 0.200000 avg loss = 0.009311
      Epoch   5 Time     31.7 lr = 0.200000 avg loss = 0.009288
      Epoch   5 Time     31.7 lr = 0.200000 avg loss = 0.009253
      Epoch   5 Time     31.5 lr = 0.200000 avg loss = 0.009232
      Epoch   5 Time     31.6 lr = 0.200000 avg loss = 0.009220
      Epoch   5 Time     31.6 lr = 0.200000 avg loss = 0.009177
      Epoch   5 Time     31.7 lr = 0.200000 avg loss = 0.009150
      Epoch   5 Time     31.5 lr = 0.200000 avg loss = 0.009110
      Epoch   5 Time     31.5 lr = 0.200000 avg loss = 0.009088
      Epoch   5 Time    405.9 lr = 0.200000 avg loss = 0.009071 accuracy = 37.66
      Epoch   6 Time     58.3 lr = 0.199795 avg loss = 0.008672
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008613
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008583
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008562
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008555
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008539
      Epoch   6 Time     31.3 lr = 0.199795 avg loss = 0.008517
      Epoch   6 Time     31.3 lr = 0.199795 avg loss = 0.008498
      Epoch   6 Time     31.6 lr = 0.199795 avg loss = 0.008483
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008474
      Epoch   6 Time     31.4 lr = 0.199795 avg loss = 0.008455
      Epoch   6 Time     31.5 lr = 0.199795 avg loss = 0.008439
      Epoch   6 Time    404.5 lr = 0.199795 avg loss = 0.008434 accuracy = 47.98
      Epoch   7 Time     58.2 lr = 0.199180 avg loss = 0.008054
      Epoch   7 Time     31.4 lr = 0.199180 avg loss = 0.008000
      Epoch   7 Time     31.5 lr = 0.199180 avg loss = 0.008015
      Epoch   7 Time     31.4 lr = 0.199180 avg loss = 0.007979
      Epoch   7 Time     31.5 lr = 0.199180 avg loss = 0.007973
      Epoch   7 Time     31.4 lr = 0.199180 avg loss = 0.007982
      Epoch   7 Time     31.5 lr = 0.199180 avg loss = 0.007975
      Epoch   7 Time     31.7 lr = 0.199180 avg loss = 0.007985
      Epoch   7 Time     31.7 lr = 0.199180 avg loss = 0.007983
      Epoch   7 Time     31.8 lr = 0.199180 avg loss = 0.007977
      Epoch   7 Time     31.5 lr = 0.199180 avg loss = 0.007962
      Epoch   7 Time     31.5 lr = 0.199180 avg loss = 0.007954
      Epoch   7 Time    405.0 lr = 0.199180 avg loss = 0.007945 accuracy = 47.02
      Epoch   8 Time     58.5 lr = 0.198158 avg loss = 0.007695
      Epoch   8 Time     31.8 lr = 0.198158 avg loss = 0.007673
      Epoch   8 Time     31.5 lr = 0.198158 avg loss = 0.007687
      Epoch   8 Time     31.8 lr = 0.198158 avg loss = 0.007671
      Epoch   8 Time     31.8 lr = 0.198158 avg loss = 0.007649
      Epoch   8 Time     31.7 lr = 0.198158 avg loss = 0.007623
      Epoch   8 Time     31.7 lr = 0.198158 avg loss = 0.007620
      Epoch   8 Time     31.6 lr = 0.198158 avg loss = 0.007606
      Epoch   8 Time     31.5 lr = 0.198158 avg loss = 0.007600
      Epoch   8 Time     31.6 lr = 0.198158 avg loss = 0.007597
      Epoch   8 Time     31.6 lr = 0.198158 avg loss = 0.007593
      Epoch   8 Time     31.4 lr = 0.198158 avg loss = 0.007569
      Epoch   8 Time    406.4 lr = 0.198158 avg loss = 0.007570 accuracy = 52.30
      Epoch   9 Time     58.3 lr = 0.196733 avg loss = 0.007204
      Epoch   9 Time     31.6 lr = 0.196733 avg loss = 0.007231
      Epoch   9 Time     31.4 lr = 0.196733 avg loss = 0.007243
      Epoch   9 Time     31.5 lr = 0.196733 avg loss = 0.007261
      Epoch   9 Time     31.4 lr = 0.196733 avg loss = 0.007256
      Epoch   9 Time     31.6 lr = 0.196733 avg loss = 0.007249
      Epoch   9 Time     31.8 lr = 0.196733 avg loss = 0.007261
      Epoch   9 Time     31.7 lr = 0.196733 avg loss = 0.007263
      Epoch   9 Time     31.7 lr = 0.196733 avg loss = 0.007264
      Epoch   9 Time     31.8 lr = 0.196733 avg loss = 0.007253
      Epoch   9 Time     31.7 lr = 0.196733 avg loss = 0.007249
      Epoch   9 Time     31.6 lr = 0.196733 avg loss = 0.007242
      Epoch   9 Time    406.5 lr = 0.196733 avg loss = 0.007247 accuracy = 50.98
      Epoch  10 Time     58.7 lr = 0.194911 avg loss = 0.006895
      Epoch  10 Time     31.6 lr = 0.194911 avg loss = 0.006943
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006968
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006986
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006992
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006996
      Epoch  10 Time     31.5 lr = 0.194911 avg loss = 0.006986
      Epoch  10 Time     31.6 lr = 0.194911 avg loss = 0.006986
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006985
      Epoch  10 Time     31.6 lr = 0.194911 avg loss = 0.006983
      Epoch  10 Time     31.8 lr = 0.194911 avg loss = 0.006983
      Epoch  10 Time     31.7 lr = 0.194911 avg loss = 0.006976
      Epoch  10 Time    407.3 lr = 0.194911 avg loss = 0.006980 accuracy = 52.56
      Epoch  11 Time     59.0 lr = 0.192699 avg loss = 0.006809
      Epoch  11 Time     31.8 lr = 0.192699 avg loss = 0.006759
      Epoch  11 Time     31.7 lr = 0.192699 avg loss = 0.006774
      Epoch  11 Time     31.4 lr = 0.192699 avg loss = 0.006792
      Epoch  11 Time     31.5 lr = 0.192699 avg loss = 0.006780
      Epoch  11 Time     31.6 lr = 0.192699 avg loss = 0.006789
      Epoch  11 Time     31.4 lr = 0.192699 avg loss = 0.006780
      Epoch  11 Time     31.5 lr = 0.192699 avg loss = 0.006788
      Epoch  11 Time     31.4 lr = 0.192699 avg loss = 0.006783
      Epoch  11 Time     31.5 lr = 0.192699 avg loss = 0.006785
      Epoch  11 Time     31.4 lr = 0.192699 avg loss = 0.006777
      Epoch  11 Time     31.4 lr = 0.192699 avg loss = 0.006775
      Epoch  11 Time    405.3 lr = 0.192699 avg loss = 0.006771 accuracy = 57.32
      Epoch  12 Time     58.4 lr = 0.190107 avg loss = 0.006601
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006563
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006585
      Epoch  12 Time     31.7 lr = 0.190107 avg loss = 0.006604
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006574
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006564
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006563
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006558
      Epoch  12 Time     31.5 lr = 0.190107 avg loss = 0.006564
      Epoch  12 Time     30.8 lr = 0.190107 avg loss = 0.006548
      Epoch  12 Time     30.8 lr = 0.190107 avg loss = 0.006556
      Epoch  12 Time     30.8 lr = 0.190107 avg loss = 0.006557
      Epoch  12 Time    402.1 lr = 0.190107 avg loss = 0.006556 accuracy = 57.40
      Epoch  13 Time     56.9 lr = 0.187145 avg loss = 0.006311
      Epoch  13 Time     30.8 lr = 0.187145 avg loss = 0.006334
      Epoch  13 Time     30.7 lr = 0.187145 avg loss = 0.006316
      Epoch  13 Time     30.6 lr = 0.187145 avg loss = 0.006341
      Epoch  13 Time     30.8 lr = 0.187145 avg loss = 0.006339
      Epoch  13 Time     30.8 lr = 0.187145 avg loss = 0.006354
      Epoch  13 Time     30.6 lr = 0.187145 avg loss = 0.006365
      Epoch  13 Time     30.8 lr = 0.187145 avg loss = 0.006381
      Epoch  13 Time     30.7 lr = 0.187145 avg loss = 0.006382
      Epoch  13 Time     30.8 lr = 0.187145 avg loss = 0.006390
      Epoch  13 Time     31.0 lr = 0.187145 avg loss = 0.006384
      Epoch  13 Time     31.4 lr = 0.187145 avg loss = 0.006380
      Epoch  13 Time    395.9 lr = 0.187145 avg loss = 0.006385 accuracy = 55.20
      Epoch  14 Time     57.4 lr = 0.183825 avg loss = 0.006127
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006186
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006213
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006200
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006230
      Epoch  14 Time     30.9 lr = 0.183825 avg loss = 0.006244
      Epoch  14 Time     30.9 lr = 0.183825 avg loss = 0.006246
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006240
      Epoch  14 Time     30.9 lr = 0.183825 avg loss = 0.006241
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006249
      Epoch  14 Time     30.8 lr = 0.183825 avg loss = 0.006242
      Epoch  14 Time     30.9 lr = 0.183825 avg loss = 0.006249
      Epoch  14 Time    396.6 lr = 0.183825 avg loss = 0.006245 accuracy = 58.56
      Epoch  15 Time     57.6 lr = 0.180161 avg loss = 0.005880
      Epoch  15 Time     30.9 lr = 0.180161 avg loss = 0.006009
      Epoch  15 Time     31.1 lr = 0.180161 avg loss = 0.006037
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006057
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006047
      Epoch  15 Time     30.7 lr = 0.180161 avg loss = 0.006053
      Epoch  15 Time     30.9 lr = 0.180161 avg loss = 0.006056
      Epoch  15 Time     30.9 lr = 0.180161 avg loss = 0.006066
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006058
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006069
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006072
      Epoch  15 Time     30.8 lr = 0.180161 avg loss = 0.006086
      Epoch  15 Time    396.7 lr = 0.180161 avg loss = 0.006090 accuracy = 55.42
      Epoch  16 Time     57.4 lr = 0.176168 avg loss = 0.005772
      Epoch  16 Time     30.9 lr = 0.176168 avg loss = 0.005850
      Epoch  16 Time     30.9 lr = 0.176168 avg loss = 0.005864
      Epoch  16 Time     31.0 lr = 0.176168 avg loss = 0.005879
      Epoch  16 Time     30.9 lr = 0.176168 avg loss = 0.005895
      Epoch  16 Time     30.8 lr = 0.176168 avg loss = 0.005904
      Epoch  16 Time     30.8 lr = 0.176168 avg loss = 0.005896
      Epoch  16 Time     31.2 lr = 0.176168 avg loss = 0.005898
      Epoch  16 Time     30.8 lr = 0.176168 avg loss = 0.005896
      Epoch  16 Time     30.8 lr = 0.176168 avg loss = 0.005904
      Epoch  16 Time     30.9 lr = 0.176168 avg loss = 0.005921
      Epoch  16 Time     30.9 lr = 0.176168 avg loss = 0.005926
      Epoch  16 Time    397.6 lr = 0.176168 avg loss = 0.005923 accuracy = 59.42
      Epoch  17 Time     58.0 lr = 0.171863 avg loss = 0.005760
      Epoch  17 Time     30.8 lr = 0.171863 avg loss = 0.005759
      Epoch  17 Time     31.0 lr = 0.171863 avg loss = 0.005769
      Epoch  17 Time     31.1 lr = 0.171863 avg loss = 0.005806
      Epoch  17 Time     31.0 lr = 0.171863 avg loss = 0.005799
      Epoch  17 Time     30.9 lr = 0.171863 avg loss = 0.005821
      Epoch  17 Time     30.9 lr = 0.171863 avg loss = 0.005811
      Epoch  17 Time     31.0 lr = 0.171863 avg loss = 0.005826
      Epoch  17 Time     31.1 lr = 0.171863 avg loss = 0.005833
      Epoch  17 Time     31.1 lr = 0.171863 avg loss = 0.005832
      Epoch  17 Time     31.0 lr = 0.171863 avg loss = 0.005840
      Epoch  17 Time     31.1 lr = 0.171863 avg loss = 0.005843
      Epoch  17 Time    398.7 lr = 0.171863 avg loss = 0.005846 accuracy = 59.68
      Epoch  18 Time     57.4 lr = 0.167263 avg loss = 0.005625
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005703
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005691
      Epoch  18 Time     31.1 lr = 0.167263 avg loss = 0.005702
      Epoch  18 Time     30.9 lr = 0.167263 avg loss = 0.005689
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005705
      Epoch  18 Time     31.2 lr = 0.167263 avg loss = 0.005703
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005714
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005709
      Epoch  18 Time     30.9 lr = 0.167263 avg loss = 0.005726
      Epoch  18 Time     31.1 lr = 0.167263 avg loss = 0.005730
      Epoch  18 Time     31.0 lr = 0.167263 avg loss = 0.005737
      Epoch  18 Time    398.8 lr = 0.167263 avg loss = 0.005732 accuracy = 58.82
      Epoch  19 Time     57.7 lr = 0.162387 avg loss = 0.005500
      Epoch  19 Time     31.0 lr = 0.162387 avg loss = 0.005504
      Epoch  19 Time     31.1 lr = 0.162387 avg loss = 0.005531
      Epoch  19 Time     31.0 lr = 0.162387 avg loss = 0.005539
      Epoch  19 Time     31.2 lr = 0.162387 avg loss = 0.005550
      Epoch  19 Time     31.1 lr = 0.162387 avg loss = 0.005548
      Epoch  19 Time     31.3 lr = 0.162387 avg loss = 0.005555
      Epoch  19 Time     31.3 lr = 0.162387 avg loss = 0.005564
      Epoch  19 Time     31.0 lr = 0.162387 avg loss = 0.005571
      Epoch  19 Time     31.2 lr = 0.162387 avg loss = 0.005580
      Epoch  19 Time     31.2 lr = 0.162387 avg loss = 0.005583
      Epoch  19 Time     31.0 lr = 0.162387 avg loss = 0.005595
      Epoch  19 Time    399.9 lr = 0.162387 avg loss = 0.005597 accuracy = 61.20
      Epoch  20 Time     57.9 lr = 0.157254 avg loss = 0.005413
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005401
      Epoch  20 Time     31.2 lr = 0.157254 avg loss = 0.005362
      Epoch  20 Time     31.2 lr = 0.157254 avg loss = 0.005399
      Epoch  20 Time     31.0 lr = 0.157254 avg loss = 0.005404
      Epoch  20 Time     31.0 lr = 0.157254 avg loss = 0.005407
      Epoch  20 Time     31.2 lr = 0.157254 avg loss = 0.005424
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005445
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005448
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005455
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005460
      Epoch  20 Time     31.1 lr = 0.157254 avg loss = 0.005469
      Epoch  20 Time    400.0 lr = 0.157254 avg loss = 0.005476 accuracy = 62.76
      Epoch  21 Time     57.9 lr = 0.151887 avg loss = 0.005202
      Epoch  21 Time     31.0 lr = 0.151887 avg loss = 0.005302
      Epoch  21 Time     31.2 lr = 0.151887 avg loss = 0.005309
      Epoch  21 Time     31.2 lr = 0.151887 avg loss = 0.005337
      Epoch  21 Time     31.1 lr = 0.151887 avg loss = 0.005348
      Epoch  21 Time     31.2 lr = 0.151887 avg loss = 0.005359
      Epoch  21 Time     31.3 lr = 0.151887 avg loss = 0.005361
      Epoch  21 Time     31.2 lr = 0.151887 avg loss = 0.005367
      Epoch  21 Time     31.1 lr = 0.151887 avg loss = 0.005378
      Epoch  21 Time     31.4 lr = 0.151887 avg loss = 0.005378
      Epoch  21 Time     31.2 lr = 0.151887 avg loss = 0.005375
      Epoch  21 Time     31.3 lr = 0.151887 avg loss = 0.005380
      Epoch  21 Time    400.9 lr = 0.151887 avg loss = 0.005383 accuracy = 61.14
      Epoch  22 Time     57.8 lr = 0.146308 avg loss = 0.005143
      Epoch  22 Time     31.2 lr = 0.146308 avg loss = 0.005130
      Epoch  22 Time     31.1 lr = 0.146308 avg loss = 0.005125
      Epoch  22 Time     31.2 lr = 0.146308 avg loss = 0.005142
      Epoch  22 Time     31.4 lr = 0.146308 avg loss = 0.005170
      Epoch  22 Time     31.2 lr = 0.146308 avg loss = 0.005193
      Epoch  22 Time     31.3 lr = 0.146308 avg loss = 0.005212
      Epoch  22 Time     31.3 lr = 0.146308 avg loss = 0.005215
      Epoch  22 Time     31.2 lr = 0.146308 avg loss = 0.005228
      Epoch  22 Time     31.4 lr = 0.146308 avg loss = 0.005241
      Epoch  22 Time     31.3 lr = 0.146308 avg loss = 0.005254
      Epoch  22 Time     31.3 lr = 0.146308 avg loss = 0.005268
      Epoch  22 Time    402.1 lr = 0.146308 avg loss = 0.005266 accuracy = 63.44
      Epoch  23 Time     58.3 lr = 0.140538 avg loss = 0.005055
      Epoch  23 Time     31.5 lr = 0.140538 avg loss = 0.005105
      Epoch  23 Time     31.1 lr = 0.140538 avg loss = 0.005114
      Epoch  23 Time     31.3 lr = 0.140538 avg loss = 0.005121
      Epoch  23 Time     31.4 lr = 0.140538 avg loss = 0.005135
      Epoch  23 Time     31.5 lr = 0.140538 avg loss = 0.005153
      Epoch  23 Time     31.3 lr = 0.140538 avg loss = 0.005152
      Epoch  23 Time     31.3 lr = 0.140538 avg loss = 0.005155
      Epoch  23 Time     31.4 lr = 0.140538 avg loss = 0.005152
      Epoch  23 Time     31.4 lr = 0.140538 avg loss = 0.005156
      Epoch  23 Time     31.4 lr = 0.140538 avg loss = 0.005161
      Epoch  23 Time     31.5 lr = 0.140538 avg loss = 0.005162
      Epoch  23 Time    403.1 lr = 0.140538 avg loss = 0.005158 accuracy = 63.50
      Epoch  24 Time     58.2 lr = 0.134602 avg loss = 0.004940
      Epoch  24 Time     31.3 lr = 0.134602 avg loss = 0.004984
      Epoch  24 Time     31.4 lr = 0.134602 avg loss = 0.005020
      Epoch  24 Time     31.5 lr = 0.134602 avg loss = 0.005018
      Epoch  24 Time     31.4 lr = 0.134602 avg loss = 0.005026
      Epoch  24 Time     31.3 lr = 0.134602 avg loss = 0.005033
      Epoch  24 Time     31.4 lr = 0.134602 avg loss = 0.005048
      Epoch  24 Time     31.3 lr = 0.134602 avg loss = 0.005062
      Epoch  24 Time     31.4 lr = 0.134602 avg loss = 0.005076
      Epoch  24 Time     31.2 lr = 0.134602 avg loss = 0.005066
      Epoch  24 Time     31.4 lr = 0.134602 avg loss = 0.005067
      Epoch  24 Time     31.3 lr = 0.134602 avg loss = 0.005074
      Epoch  24 Time    403.3 lr = 0.134602 avg loss = 0.005074 accuracy = 64.10
      Epoch  25 Time     58.3 lr = 0.128524 avg loss = 0.004893
      Epoch  25 Time     31.4 lr = 0.128524 avg loss = 0.004862
      Epoch  25 Time     31.3 lr = 0.128524 avg loss = 0.004878
      Epoch  25 Time     31.5 lr = 0.128524 avg loss = 0.004885
      Epoch  25 Time     31.5 lr = 0.128524 avg loss = 0.004899
      Epoch  25 Time     31.5 lr = 0.128524 avg loss = 0.004924
      Epoch  25 Time     31.4 lr = 0.128524 avg loss = 0.004938
      Epoch  25 Time     31.5 lr = 0.128524 avg loss = 0.004935
      Epoch  25 Time     31.4 lr = 0.128524 avg loss = 0.004932
      Epoch  25 Time     31.6 lr = 0.128524 avg loss = 0.004941
      Epoch  25 Time     31.3 lr = 0.128524 avg loss = 0.004942
      Epoch  25 Time     31.3 lr = 0.128524 avg loss = 0.004952
      Epoch  25 Time    403.9 lr = 0.128524 avg loss = 0.004952 accuracy = 62.66
      Epoch  26 Time     58.3 lr = 0.122330 avg loss = 0.004806
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004793
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004791
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004801
      Epoch  26 Time     31.5 lr = 0.122330 avg loss = 0.004795
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004806
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004825
      Epoch  26 Time     31.5 lr = 0.122330 avg loss = 0.004844
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004844
      Epoch  26 Time     31.3 lr = 0.122330 avg loss = 0.004856
      Epoch  26 Time     31.4 lr = 0.122330 avg loss = 0.004862
      Epoch  26 Time     31.5 lr = 0.122330 avg loss = 0.004863
      Epoch  26 Time    404.1 lr = 0.122330 avg loss = 0.004865 accuracy = 65.32
      Epoch  27 Time     58.4 lr = 0.116044 avg loss = 0.004710
      Epoch  27 Time     31.3 lr = 0.116044 avg loss = 0.004755
      Epoch  27 Time     31.2 lr = 0.116044 avg loss = 0.004726
      Epoch  27 Time     31.5 lr = 0.116044 avg loss = 0.004710
      Epoch  27 Time     31.4 lr = 0.116044 avg loss = 0.004736
      Epoch  27 Time     31.5 lr = 0.116044 avg loss = 0.004730
      Epoch  27 Time     31.5 lr = 0.116044 avg loss = 0.004730
      Epoch  27 Time     31.4 lr = 0.116044 avg loss = 0.004740
      Epoch  27 Time     31.4 lr = 0.116044 avg loss = 0.004739
      Epoch  27 Time     31.5 lr = 0.116044 avg loss = 0.004745
      Epoch  27 Time     31.4 lr = 0.116044 avg loss = 0.004748
      Epoch  27 Time     31.5 lr = 0.116044 avg loss = 0.004755
      Epoch  27 Time    403.8 lr = 0.116044 avg loss = 0.004758 accuracy = 66.26
      Epoch  28 Time     58.4 lr = 0.109693 avg loss = 0.004566
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004620
      Epoch  28 Time     31.3 lr = 0.109693 avg loss = 0.004600
      Epoch  28 Time     31.5 lr = 0.109693 avg loss = 0.004587
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004614
      Epoch  28 Time     31.2 lr = 0.109693 avg loss = 0.004626
      Epoch  28 Time     31.5 lr = 0.109693 avg loss = 0.004624
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004638
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004635
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004642
      Epoch  28 Time     31.4 lr = 0.109693 avg loss = 0.004646
      Epoch  28 Time     31.5 lr = 0.109693 avg loss = 0.004659
      Epoch  28 Time    403.9 lr = 0.109693 avg loss = 0.004667 accuracy = 63.98
      Epoch  29 Time     58.6 lr = 0.103302 avg loss = 0.004522
      Epoch  29 Time     31.7 lr = 0.103302 avg loss = 0.004539
      Epoch  29 Time     31.5 lr = 0.103302 avg loss = 0.004533
      Epoch  29 Time     31.7 lr = 0.103302 avg loss = 0.004541
      Epoch  29 Time     31.4 lr = 0.103302 avg loss = 0.004531
      Epoch  29 Time     31.5 lr = 0.103302 avg loss = 0.004529
      Epoch  29 Time     31.6 lr = 0.103302 avg loss = 0.004546
      Epoch  29 Time     31.4 lr = 0.103302 avg loss = 0.004540
      Epoch  29 Time     31.5 lr = 0.103302 avg loss = 0.004532
      Epoch  29 Time     31.5 lr = 0.103302 avg loss = 0.004526
      Epoch  29 Time     31.5 lr = 0.103302 avg loss = 0.004539
      Epoch  29 Time     31.4 lr = 0.103302 avg loss = 0.004547
      Epoch  29 Time    405.6 lr = 0.103302 avg loss = 0.004542 accuracy = 64.26
      Epoch  30 Time     58.8 lr = 0.096898 avg loss = 0.004402
      Epoch  30 Time     31.6 lr = 0.096898 avg loss = 0.004438
      Epoch  30 Time     31.7 lr = 0.096898 avg loss = 0.004431
      Epoch  30 Time     31.4 lr = 0.096898 avg loss = 0.004437
      Epoch  30 Time     31.6 lr = 0.096898 avg loss = 0.004446
      Epoch  30 Time     31.6 lr = 0.096898 avg loss = 0.004439
      Epoch  30 Time     31.8 lr = 0.096898 avg loss = 0.004434
      Epoch  30 Time     31.5 lr = 0.096898 avg loss = 0.004441
      Epoch  30 Time     31.5 lr = 0.096898 avg loss = 0.004443
      Epoch  30 Time     31.5 lr = 0.096898 avg loss = 0.004452
      Epoch  30 Time     31.5 lr = 0.096898 avg loss = 0.004450
      Epoch  30 Time     31.6 lr = 0.096898 avg loss = 0.004458
      Epoch  30 Time    406.1 lr = 0.096898 avg loss = 0.004463 accuracy = 67.46
      Epoch  31 Time     58.8 lr = 0.090507 avg loss = 0.004235
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004229
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004314
      Epoch  31 Time     31.6 lr = 0.090507 avg loss = 0.004288
      Epoch  31 Time     31.6 lr = 0.090507 avg loss = 0.004294
      Epoch  31 Time     31.6 lr = 0.090507 avg loss = 0.004301
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004310
      Epoch  31 Time     31.6 lr = 0.090507 avg loss = 0.004314
      Epoch  31 Time     31.6 lr = 0.090507 avg loss = 0.004322
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004336
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004339
      Epoch  31 Time     31.5 lr = 0.090507 avg loss = 0.004350
      Epoch  31 Time    405.9 lr = 0.090507 avg loss = 0.004351 accuracy = 68.82
      Epoch  32 Time     58.8 lr = 0.084156 avg loss = 0.004262
      Epoch  32 Time     31.5 lr = 0.084156 avg loss = 0.004224
      Epoch  32 Time     31.4 lr = 0.084156 avg loss = 0.004220
      Epoch  32 Time     31.4 lr = 0.084156 avg loss = 0.004229
      Epoch  32 Time     31.6 lr = 0.084156 avg loss = 0.004226
      Epoch  32 Time     31.4 lr = 0.084156 avg loss = 0.004235
      Epoch  32 Time     31.6 lr = 0.084156 avg loss = 0.004252
      Epoch  32 Time     31.7 lr = 0.084156 avg loss = 0.004249
      Epoch  32 Time     31.9 lr = 0.084156 avg loss = 0.004244
      Epoch  32 Time     31.2 lr = 0.084156 avg loss = 0.004249
      Epoch  32 Time     31.6 lr = 0.084156 avg loss = 0.004250
      Epoch  32 Time     31.5 lr = 0.084156 avg loss = 0.004262
      Epoch  32 Time    405.2 lr = 0.084156 avg loss = 0.004268 accuracy = 66.18
      Epoch  33 Time     58.8 lr = 0.077870 avg loss = 0.004103
      Epoch  33 Time     31.4 lr = 0.077870 avg loss = 0.004114
      Epoch  33 Time     31.6 lr = 0.077870 avg loss = 0.004122
      Epoch  33 Time     31.5 lr = 0.077870 avg loss = 0.004114
      Epoch  33 Time     31.6 lr = 0.077870 avg loss = 0.004113
      Epoch  33 Time     31.7 lr = 0.077870 avg loss = 0.004117
      Epoch  33 Time     31.8 lr = 0.077870 avg loss = 0.004121
      Epoch  33 Time     31.6 lr = 0.077870 avg loss = 0.004132
      Epoch  33 Time     31.5 lr = 0.077870 avg loss = 0.004144
      Epoch  33 Time     31.8 lr = 0.077870 avg loss = 0.004148
      Epoch  33 Time     31.6 lr = 0.077870 avg loss = 0.004145
      Epoch  33 Time     31.6 lr = 0.077870 avg loss = 0.004149
      Epoch  33 Time    406.3 lr = 0.077870 avg loss = 0.004152 accuracy = 69.02
      Epoch  34 Time     59.2 lr = 0.071676 avg loss = 0.003950
      Epoch  34 Time     31.6 lr = 0.071676 avg loss = 0.003935
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.003984
      Epoch  34 Time     31.6 lr = 0.071676 avg loss = 0.004009
      Epoch  34 Time     31.7 lr = 0.071676 avg loss = 0.004011
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.004012
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.004014
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.004018
      Epoch  34 Time     31.4 lr = 0.071676 avg loss = 0.004031
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.004031
      Epoch  34 Time     31.5 lr = 0.071676 avg loss = 0.004038
      Epoch  34 Time     31.6 lr = 0.071676 avg loss = 0.004052
      Epoch  34 Time    406.5 lr = 0.071676 avg loss = 0.004052 accuracy = 69.02
      Epoch  35 Time     59.0 lr = 0.065598 avg loss = 0.003858
      Epoch  35 Time     31.8 lr = 0.065598 avg loss = 0.003833
      Epoch  35 Time     31.5 lr = 0.065598 avg loss = 0.003834
      Epoch  35 Time     31.7 lr = 0.065598 avg loss = 0.003849
      Epoch  35 Time     31.7 lr = 0.065598 avg loss = 0.003850
      Epoch  35 Time     31.6 lr = 0.065598 avg loss = 0.003870
      Epoch  35 Time     31.5 lr = 0.065598 avg loss = 0.003880
      Epoch  35 Time     31.4 lr = 0.065598 avg loss = 0.003888
      Epoch  35 Time     31.4 lr = 0.065598 avg loss = 0.003887
      Epoch  35 Time     31.5 lr = 0.065598 avg loss = 0.003909
      Epoch  35 Time     31.5 lr = 0.065598 avg loss = 0.003918
      Epoch  35 Time     31.4 lr = 0.065598 avg loss = 0.003929
      Epoch  35 Time    405.7 lr = 0.065598 avg loss = 0.003933 accuracy = 67.24
      Epoch  36 Time     58.5 lr = 0.059662 avg loss = 0.003806
      Epoch  36 Time     31.7 lr = 0.059662 avg loss = 0.003802
      Epoch  36 Time     31.5 lr = 0.059662 avg loss = 0.003811
      Epoch  36 Time     31.7 lr = 0.059662 avg loss = 0.003816
      Epoch  36 Time     31.4 lr = 0.059662 avg loss = 0.003814
      Epoch  36 Time     31.3 lr = 0.059662 avg loss = 0.003818
      Epoch  36 Time     31.3 lr = 0.059662 avg loss = 0.003810
      Epoch  36 Time     31.3 lr = 0.059662 avg loss = 0.003813
      Epoch  36 Time     31.6 lr = 0.059662 avg loss = 0.003817
      Epoch  36 Time     31.3 lr = 0.059662 avg loss = 0.003820
      Epoch  36 Time     31.4 lr = 0.059662 avg loss = 0.003823
      Epoch  36 Time     31.3 lr = 0.059662 avg loss = 0.003831
      Epoch  36 Time    404.2 lr = 0.059662 avg loss = 0.003837 accuracy = 69.40
      Epoch  37 Time     58.3 lr = 0.053892 avg loss = 0.003751
      Epoch  37 Time     31.5 lr = 0.053892 avg loss = 0.003702
      Epoch  37 Time     31.5 lr = 0.053892 avg loss = 0.003661
      Epoch  37 Time     31.7 lr = 0.053892 avg loss = 0.003678
      Epoch  37 Time     31.2 lr = 0.053892 avg loss = 0.003682
      Epoch  37 Time     31.3 lr = 0.053892 avg loss = 0.003679
      Epoch  37 Time     31.4 lr = 0.053892 avg loss = 0.003689
      Epoch  37 Time     31.3 lr = 0.053892 avg loss = 0.003685
      Epoch  37 Time     31.3 lr = 0.053892 avg loss = 0.003680
      Epoch  37 Time     31.5 lr = 0.053892 avg loss = 0.003681
      Epoch  37 Time     31.4 lr = 0.053892 avg loss = 0.003684
      Epoch  37 Time     31.3 lr = 0.053892 avg loss = 0.003694
      Epoch  37 Time    404.0 lr = 0.053892 avg loss = 0.003706 accuracy = 69.04
      Epoch  38 Time     58.5 lr = 0.048313 avg loss = 0.003563
      Epoch  38 Time     31.4 lr = 0.048313 avg loss = 0.003557
      Epoch  38 Time     31.5 lr = 0.048313 avg loss = 0.003582
      Epoch  38 Time     31.3 lr = 0.048313 avg loss = 0.003592
      Epoch  38 Time     31.3 lr = 0.048313 avg loss = 0.003596
      Epoch  38 Time     31.5 lr = 0.048313 avg loss = 0.003585
      Epoch  38 Time     31.5 lr = 0.048313 avg loss = 0.003600
      Epoch  38 Time     31.5 lr = 0.048313 avg loss = 0.003617
      Epoch  38 Time     31.4 lr = 0.048313 avg loss = 0.003603
      Epoch  38 Time     31.4 lr = 0.048313 avg loss = 0.003601
      Epoch  38 Time     31.5 lr = 0.048313 avg loss = 0.003607
      Epoch  38 Time     31.4 lr = 0.048313 avg loss = 0.003614
      Epoch  38 Time    404.0 lr = 0.048313 avg loss = 0.003627 accuracy = 70.60
      Epoch  39 Time     58.4 lr = 0.042946 avg loss = 0.003408
      Epoch  39 Time     31.6 lr = 0.042946 avg loss = 0.003427
      Epoch  39 Time     31.5 lr = 0.042946 avg loss = 0.003440
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003474
      Epoch  39 Time     31.6 lr = 0.042946 avg loss = 0.003496
      Epoch  39 Time     31.5 lr = 0.042946 avg loss = 0.003493
      Epoch  39 Time     31.5 lr = 0.042946 avg loss = 0.003482
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003485
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003491
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003501
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003501
      Epoch  39 Time     31.4 lr = 0.042946 avg loss = 0.003498
      Epoch  39 Time    404.7 lr = 0.042946 avg loss = 0.003507 accuracy = 69.32
      Epoch  40 Time     58.3 lr = 0.037813 avg loss = 0.003398
      Epoch  40 Time     31.3 lr = 0.037813 avg loss = 0.003401
      Epoch  40 Time     31.6 lr = 0.037813 avg loss = 0.003391
      Epoch  40 Time     31.3 lr = 0.037813 avg loss = 0.003379
      Epoch  40 Time     31.5 lr = 0.037813 avg loss = 0.003367
      Epoch  40 Time     31.5 lr = 0.037813 avg loss = 0.003342
      Epoch  40 Time     31.4 lr = 0.037813 avg loss = 0.003345
      Epoch  40 Time     31.4 lr = 0.037813 avg loss = 0.003354
      Epoch  40 Time     31.4 lr = 0.037813 avg loss = 0.003369
      Epoch  40 Time     31.5 lr = 0.037813 avg loss = 0.003371
      Epoch  40 Time     31.4 lr = 0.037813 avg loss = 0.003369
      Epoch  40 Time     31.4 lr = 0.037813 avg loss = 0.003379
      Epoch  40 Time    404.2 lr = 0.037813 avg loss = 0.003380 accuracy = 71.36
      Epoch  41 Time     58.8 lr = 0.032937 avg loss = 0.003228
      Epoch  41 Time     31.7 lr = 0.032937 avg loss = 0.003272
      Epoch  41 Time     31.3 lr = 0.032937 avg loss = 0.003278
      Epoch  41 Time     31.4 lr = 0.032937 avg loss = 0.003256
      Epoch  41 Time     31.6 lr = 0.032937 avg loss = 0.003271
      Epoch  41 Time     31.3 lr = 0.032937 avg loss = 0.003280
      Epoch  41 Time     31.4 lr = 0.032937 avg loss = 0.003289
      Epoch  41 Time     31.3 lr = 0.032937 avg loss = 0.003286
      Epoch  41 Time     31.6 lr = 0.032937 avg loss = 0.003289
      Epoch  41 Time     31.4 lr = 0.032937 avg loss = 0.003293
      Epoch  41 Time     31.4 lr = 0.032937 avg loss = 0.003296
      Epoch  41 Time     31.4 lr = 0.032937 avg loss = 0.003292
      Epoch  41 Time    404.5 lr = 0.032937 avg loss = 0.003299 accuracy = 71.76
      Epoch  42 Time     58.5 lr = 0.028337 avg loss = 0.003244
      Epoch  42 Time     31.5 lr = 0.028337 avg loss = 0.003183
      Epoch  42 Time     31.4 lr = 0.028337 avg loss = 0.003172
      Epoch  42 Time     31.7 lr = 0.028337 avg loss = 0.003148
      Epoch  42 Time     31.6 lr = 0.028337 avg loss = 0.003142
      Epoch  42 Time     31.6 lr = 0.028337 avg loss = 0.003145
      Epoch  42 Time     31.6 lr = 0.028337 avg loss = 0.003159
      Epoch  42 Time     31.6 lr = 0.028337 avg loss = 0.003166
      Epoch  42 Time     31.7 lr = 0.028337 avg loss = 0.003171
      Epoch  42 Time     31.5 lr = 0.028337 avg loss = 0.003176
      Epoch  42 Time     31.4 lr = 0.028337 avg loss = 0.003177
      Epoch  42 Time     31.4 lr = 0.028337 avg loss = 0.003172
      Epoch  42 Time    405.4 lr = 0.028337 avg loss = 0.003174 accuracy = 71.12
      Epoch  43 Time     58.6 lr = 0.024032 avg loss = 0.003132
      Epoch  43 Time     31.4 lr = 0.024032 avg loss = 0.003076
      Epoch  43 Time     31.6 lr = 0.024032 avg loss = 0.003056
      Epoch  43 Time     31.6 lr = 0.024032 avg loss = 0.003053
      Epoch  43 Time     31.5 lr = 0.024032 avg loss = 0.003058
      Epoch  43 Time     31.4 lr = 0.024032 avg loss = 0.003086
      Epoch  43 Time     31.6 lr = 0.024032 avg loss = 0.003084
      Epoch  43 Time     31.3 lr = 0.024032 avg loss = 0.003089
      Epoch  43 Time     31.4 lr = 0.024032 avg loss = 0.003094
      Epoch  43 Time     31.4 lr = 0.024032 avg loss = 0.003098
      Epoch  43 Time     31.4 lr = 0.024032 avg loss = 0.003095
      Epoch  43 Time     31.5 lr = 0.024032 avg loss = 0.003094
      Epoch  43 Time    404.7 lr = 0.024032 avg loss = 0.003099 accuracy = 71.80
      Epoch  44 Time     58.5 lr = 0.020039 avg loss = 0.003019
      Epoch  44 Time     31.6 lr = 0.020039 avg loss = 0.003011
      Epoch  44 Time     31.8 lr = 0.020039 avg loss = 0.002997
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.002991
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.003003
      Epoch  44 Time     31.3 lr = 0.020039 avg loss = 0.002996
      Epoch  44 Time     31.7 lr = 0.020039 avg loss = 0.002994
      Epoch  44 Time     31.3 lr = 0.020039 avg loss = 0.002999
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.003001
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.002995
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.003003
      Epoch  44 Time     31.5 lr = 0.020039 avg loss = 0.003005
      Epoch  44 Time    405.1 lr = 0.020039 avg loss = 0.003005 accuracy = 72.40
      Epoch  45 Time     58.3 lr = 0.016375 avg loss = 0.002971
      Epoch  45 Time     31.4 lr = 0.016375 avg loss = 0.002941
      Epoch  45 Time     31.7 lr = 0.016375 avg loss = 0.002913
      Epoch  45 Time     31.7 lr = 0.016375 avg loss = 0.002922
      Epoch  45 Time     31.7 lr = 0.016375 avg loss = 0.002910
      Epoch  45 Time     31.4 lr = 0.016375 avg loss = 0.002900
      Epoch  45 Time     31.2 lr = 0.016375 avg loss = 0.002893
      Epoch  45 Time     31.4 lr = 0.016375 avg loss = 0.002897
      Epoch  45 Time     31.3 lr = 0.016375 avg loss = 0.002903
      Epoch  45 Time     31.4 lr = 0.016375 avg loss = 0.002905
      Epoch  45 Time     31.3 lr = 0.016375 avg loss = 0.002911
      Epoch  45 Time     31.1 lr = 0.016375 avg loss = 0.002913
      Epoch  45 Time    403.8 lr = 0.016375 avg loss = 0.002911 accuracy = 71.62
      Epoch  46 Time     58.0 lr = 0.013055 avg loss = 0.002764
      Epoch  46 Time     31.2 lr = 0.013055 avg loss = 0.002777
      Epoch  46 Time     31.2 lr = 0.013055 avg loss = 0.002790
      Epoch  46 Time     31.1 lr = 0.013055 avg loss = 0.002811
      Epoch  46 Time     31.2 lr = 0.013055 avg loss = 0.002808
      Epoch  46 Time     31.1 lr = 0.013055 avg loss = 0.002803
      Epoch  46 Time     31.2 lr = 0.013055 avg loss = 0.002805
      Epoch  46 Time     31.0 lr = 0.013055 avg loss = 0.002807
      Epoch  46 Time     31.1 lr = 0.013055 avg loss = 0.002811
      Epoch  46 Time     31.2 lr = 0.013055 avg loss = 0.002814
      Epoch  46 Time     31.3 lr = 0.013055 avg loss = 0.002820
      Epoch  46 Time     31.1 lr = 0.013055 avg loss = 0.002826
      Epoch  46 Time    400.8 lr = 0.013055 avg loss = 0.002828 accuracy = 72.26
      Epoch  47 Time     58.0 lr = 0.010093 avg loss = 0.002738
      Epoch  47 Time     31.2 lr = 0.010093 avg loss = 0.002752
      Epoch  47 Time     31.1 lr = 0.010093 avg loss = 0.002769
      Epoch  47 Time     31.1 lr = 0.010093 avg loss = 0.002760
      Epoch  47 Time     31.1 lr = 0.010093 avg loss = 0.002748
      Epoch  47 Time     31.0 lr = 0.010093 avg loss = 0.002736
      Epoch  47 Time     31.2 lr = 0.010093 avg loss = 0.002739
      Epoch  47 Time     31.3 lr = 0.010093 avg loss = 0.002747
      Epoch  47 Time     31.0 lr = 0.010093 avg loss = 0.002759
      Epoch  47 Time     31.2 lr = 0.010093 avg loss = 0.002760
      Epoch  47 Time     31.2 lr = 0.010093 avg loss = 0.002756
      Epoch  47 Time     31.1 lr = 0.010093 avg loss = 0.002752
      Epoch  47 Time    400.8 lr = 0.010093 avg loss = 0.002749 accuracy = 72.92
      Epoch  48 Time     58.0 lr = 0.007501 avg loss = 0.002726
      Epoch  48 Time     31.0 lr = 0.007501 avg loss = 0.002705
      Epoch  48 Time     31.0 lr = 0.007501 avg loss = 0.002674
      Epoch  48 Time     31.1 lr = 0.007501 avg loss = 0.002647
      Epoch  48 Time     31.0 lr = 0.007501 avg loss = 0.002665
      Epoch  48 Time     31.1 lr = 0.007501 avg loss = 0.002659
      Epoch  48 Time     31.0 lr = 0.007501 avg loss = 0.002667
      Epoch  48 Time     31.0 lr = 0.007501 avg loss = 0.002668
      Epoch  48 Time     31.1 lr = 0.007501 avg loss = 0.002660
      Epoch  48 Time     31.2 lr = 0.007501 avg loss = 0.002664
      Epoch  48 Time     31.1 lr = 0.007501 avg loss = 0.002668
      Epoch  48 Time     31.2 lr = 0.007501 avg loss = 0.002668
      Epoch  48 Time    399.5 lr = 0.007501 avg loss = 0.002671 accuracy = 73.00
      Epoch  49 Time     57.9 lr = 0.005289 avg loss = 0.002685
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002663
      Epoch  49 Time     31.2 lr = 0.005289 avg loss = 0.002642
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002654
      Epoch  49 Time     31.0 lr = 0.005289 avg loss = 0.002639
      Epoch  49 Time     31.3 lr = 0.005289 avg loss = 0.002644
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002645
      Epoch  49 Time     31.2 lr = 0.005289 avg loss = 0.002635
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002629
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002622
      Epoch  49 Time     31.1 lr = 0.005289 avg loss = 0.002623
      Epoch  49 Time     31.0 lr = 0.005289 avg loss = 0.002621
      Epoch  49 Time    399.9 lr = 0.005289 avg loss = 0.002618 accuracy = 72.68
      Epoch  50 Time     57.8 lr = 0.003467 avg loss = 0.002583
      Epoch  50 Time     31.0 lr = 0.003467 avg loss = 0.002574
      Epoch  50 Time     31.2 lr = 0.003467 avg loss = 0.002580
      Epoch  50 Time     31.2 lr = 0.003467 avg loss = 0.002604
      Epoch  50 Time     31.1 lr = 0.003467 avg loss = 0.002607
      Epoch  50 Time     31.1 lr = 0.003467 avg loss = 0.002605
      Epoch  50 Time     31.1 lr = 0.003467 avg loss = 0.002609
      Epoch  50 Time     31.0 lr = 0.003467 avg loss = 0.002609
      Epoch  50 Time     31.1 lr = 0.003467 avg loss = 0.002609
      Epoch  50 Time     31.0 lr = 0.003467 avg loss = 0.002602
      Epoch  50 Time     31.0 lr = 0.003467 avg loss = 0.002599
      Epoch  50 Time     31.3 lr = 0.003467 avg loss = 0.002597
      Epoch  50 Time    399.9 lr = 0.003467 avg loss = 0.002602 accuracy = 72.96
      Epoch  51 Time     57.7 lr = 0.002042 avg loss = 0.002567
      Epoch  51 Time     31.1 lr = 0.002042 avg loss = 0.002568
      Epoch  51 Time     30.9 lr = 0.002042 avg loss = 0.002555
      Epoch  51 Time     31.1 lr = 0.002042 avg loss = 0.002550
      Epoch  51 Time     31.3 lr = 0.002042 avg loss = 0.002562
      Epoch  51 Time     31.1 lr = 0.002042 avg loss = 0.002580
      Epoch  51 Time     31.0 lr = 0.002042 avg loss = 0.002577
      Epoch  51 Time     31.1 lr = 0.002042 avg loss = 0.002567
      Epoch  51 Time     30.9 lr = 0.002042 avg loss = 0.002561
      Epoch  51 Time     31.0 lr = 0.002042 avg loss = 0.002555
      Epoch  51 Time     31.0 lr = 0.002042 avg loss = 0.002557
      Epoch  51 Time     31.0 lr = 0.002042 avg loss = 0.002553
      Epoch  51 Time    399.1 lr = 0.002042 avg loss = 0.002552 accuracy = 73.42
      Epoch  52 Time     57.5 lr = 0.001020 avg loss = 0.002560
      Epoch  52 Time     31.0 lr = 0.001020 avg loss = 0.002589
      Epoch  52 Time     30.8 lr = 0.001020 avg loss = 0.002564
      Epoch  52 Time     30.7 lr = 0.001020 avg loss = 0.002566
      Epoch  52 Time     31.2 lr = 0.001020 avg loss = 0.002561
      Epoch  52 Time     30.8 lr = 0.001020 avg loss = 0.002560
      Epoch  52 Time     30.9 lr = 0.001020 avg loss = 0.002561
      Epoch  52 Time     30.9 lr = 0.001020 avg loss = 0.002552
      Epoch  52 Time     31.0 lr = 0.001020 avg loss = 0.002551
      Epoch  52 Time     30.7 lr = 0.001020 avg loss = 0.002548
      Epoch  52 Time     30.9 lr = 0.001020 avg loss = 0.002550
      Epoch  52 Time     30.9 lr = 0.001020 avg loss = 0.002553
      Epoch  52 Time    397.3 lr = 0.001020 avg loss = 0.002551 accuracy = 73.72
      Epoch  53 Time     57.2 lr = 0.000405 avg loss = 0.002537
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002504
      Epoch  53 Time     31.0 lr = 0.000405 avg loss = 0.002535
      Epoch  53 Time     31.0 lr = 0.000405 avg loss = 0.002521
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002523
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002525
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002526
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002531
      Epoch  53 Time     30.7 lr = 0.000405 avg loss = 0.002521
      Epoch  53 Time     30.7 lr = 0.000405 avg loss = 0.002520
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002522
      Epoch  53 Time     30.8 lr = 0.000405 avg loss = 0.002520
      Epoch  53 Time    396.2 lr = 0.000405 avg loss = 0.002523 accuracy = 73.28
      Epoch  54 Time     57.2 lr = 0.000200 avg loss = 0.002479
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002492
      Epoch  54 Time     30.8 lr = 0.000200 avg loss = 0.002520
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002530
      Epoch  54 Time     30.8 lr = 0.000200 avg loss = 0.002520
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002521
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002529
      Epoch  54 Time     30.8 lr = 0.000200 avg loss = 0.002520
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002523
      Epoch  54 Time     30.6 lr = 0.000200 avg loss = 0.002519
      Epoch  54 Time     30.9 lr = 0.000200 avg loss = 0.002514
      Epoch  54 Time     31.0 lr = 0.000200 avg loss = 0.002520
      Epoch  54 Time    396.4 lr = 0.000200 avg loss = 0.002521 accuracy = 73.46

