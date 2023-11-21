[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)

# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

python parallel_check.py output:
MAP

================================================================================
Parallel Accelerator Optimizing: Function tensor_map.<locals>.\_map,
/content/mle-module-3-Sidv2001/minitorch/fast_ops.py (154)  
================================================================================

Parallel loop listing for Function tensor_map.<locals>.\_map, /content/mle-module-3-Sidv2001/minitorch/fast_ops.py (154)
-------------------------------------------------------------------------|loop #ID
def \_map( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
in_storage: Storage, |
in_shape: Shape, |
in_strides: Strides, |
) -> None: | # TODO: Implement for Task 3.1. |
for o in prange(len(out)):---------------------------------------| #0
out_index: Index = np.zeros_like(out_shape) |
in_index: Index = np.zeros_like(in_shape) |
to_index(o, out_shape, out_index) |
broadcast_index(out_index, out_shape, in_shape, in_index) |
i = index_to_position(in_index, in_strides) |
out[o] = fn(in_storage[i]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).

---

## ----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
Parallel Accelerator Optimizing: Function tensor_zip.<locals>.\_zip,
/content/mle-module-3-Sidv2001/minitorch/fast_ops.py (196)  
================================================================================

Parallel loop listing for Function tensor_zip.<locals>.\_zip, /content/mle-module-3-Sidv2001/minitorch/fast_ops.py (196)
-------------------------------------------------------------------------------|loop #ID
def \_zip( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
b_storage: Storage, |
b_shape: Shape, |
b_strides: Strides, |
) -> None: | # TODO: Implement for Task 3.1. |
for o in prange(len(out)): # prange is used for parallel execution----| #1
out_index = np.zeros_like(out_shape) |
a_index = np.zeros_like(a_shape) |
b_index = np.zeros_like(b_shape) |
to_index(o, out_shape, out_index) |
broadcast_index(out_index, out_shape, a_shape, a_index) |
broadcast_index(out_index, out_shape, b_shape, b_index) |
a_pos = index_to_position(a_index, a_strides) |
b_pos = index_to_position(b_index, b_strides) |
out[o] = fn(a_storage[a_pos], b_storage[b_pos]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).

---

## ----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
Parallel Accelerator Optimizing: Function tensor_reduce.<locals>.\_reduce,
/content/mle-module-3-Sidv2001/minitorch/fast_ops.py (241)  
================================================================================

Parallel loop listing for Function tensor_reduce.<locals>.\_reduce, /content/mle-module-3-Sidv2001/minitorch/fast_ops.py (241)
-------------------------------------------------------------|loop #ID
def \_reduce( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
reduce_dim: int, |
) -> None: | # TODO: Implement for Task 3.1. |
|
for o in prange(len(out)):---------------------------| #2
out_index: Index = np.zeros_like(out_shape) |
reduce_size = a_shape[reduce_dim] |
to_index(o, out_shape, out_index) |
res = index_to_position(out_index, a_strides) |
dim_stride = a_strides[reduce_dim] |
for s in range(reduce_size): |
j = res + (s \* dim_stride) |
out[o] = fn(out[o], a_storage[j]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).

---

## ----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
Parallel Accelerator Optimizing: Function \_tensor_matrix_multiply,
/content/mle-module-3-Sidv2001/minitorch/fast_ops.py (265)  
================================================================================

Parallel loop listing for Function \_tensor\*matrix_multiply, /content/mle-module-3-Sidv2001/minitorch/fast_ops.py (265)
------------------------------------------------------------------------------------------------------------------|loop #ID
def \_tensor_matrix_multiply( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
b_storage: Storage, |
b_shape: Shape, |
b_strides: Strides, |
) -> None: |
""" |
NUMBA tensor matrix multiply function. |
|
Should work for any tensor shapes that broadcast as long as |
|
`                                                                                                       | 
    assert a_shape[-1] == b_shape[-2]                                                                             | 
` |
|
Optimizations: |
|

- Outer loop in parallel |
  _ No index buffers or function calls |
  _ Inner loop should have no global writes, 1 multiply. |
  |
  |
  Args: |
  out (Storage): storage for `out` tensor |
  out*shape (Shape): shape for `out` tensor |
  out_strides (Strides): strides for `out` tensor |
  a_storage (Storage): storage for `a` tensor |
  a_shape (Shape): shape for `a` tensor |
  a_strides (Strides): strides for `a` tensor |
  b_storage (Storage): storage for `b` tensor |
  b_shape (Shape): shape for `b` tensor |
  b_strides (Strides): strides for `b` tensor |
  |
  Returns: |
  None : Fills in `out` |
  """ |
  a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0 |
  b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0 |
  |
  | # TODO: Implement for Task 3.2. |
  assert a_shape[-1] == b_shape[-2] |
  for o in prange(len(out)):------------------------------------------------------------------------------------| #3 # o_batch_id = o // out_strides[0] | # o_row_id = (o % out_strides[0]) // out_strides[1] | # o_col_id = o % out_shape[-1] |
  o_batch_id = o // (out_shape[-1] * out*shape[-2]) |
  out_row_id = (o % (out_shape[-1] * out*shape[-2])) // out_shape[-1] |
  out_col_id = o % out_shape[-1] |
  a_batch_row = o_batch_id * a*batch_stride + out_row_id * a*strides[-2] |
  b_batch_col = o_batch_id * b*batch_stride + out_col_id * b*strides[-1] |
  res = 0 |
  for i in range(a_shape[-1]): |
  res += a_storage[a_batch_row + (i * a*strides[-1])] * b*storage[b_batch_col + (i * b_strides[-2])] |
  out[o] = res |
  --------------------------------- Fusing loops ---------------------------------
  Attempting fusion of parallel loops (combines loops with similar properties)...
  Following the attempted fusion of parallel for-loops there are 1 parallel for-
  loop(s) (originating from loops labelled: #3).

---

## ----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

Cuda Matrix Multiply grpah:
![Alt text](<Screenshot 2023-11-21 at 5.45.37 PM.png>)

Answers for 3.5:
Split:
CPU:
Epoch 0 loss 8.289694444686944 correct 25 Time Taken: 17.606080055236816
Epoch 10 loss 6.177479436690812 correct 38 Time Taken: 0.16226983070373535
Epoch 20 loss 5.954786880276633 correct 40 Time Taken: 0.2830886125564575
Epoch 30 loss 4.635780826904739 correct 44 Time Taken: 0.2977741718292236
Epoch 40 loss 3.3481247154759557 correct 41 Time Taken: 0.31576406955718994
Epoch 50 loss 3.004030719085276 correct 47 Time Taken: 0.34809436798095705
Epoch 60 loss 4.443795534603427 correct 46 Time Taken: 0.2364044427871704
Epoch 70 loss 1.4496592376748627 correct 48 Time Taken: 0.15151093006134034
Epoch 80 loss 2.4994287844657617 correct 49 Time Taken: 0.15043239593505858
Epoch 90 loss 2.2371131318770976 correct 50 Time Taken: 0.16246888637542725
Epoch 100 loss 1.755756125489781 correct 49 Time Taken: 0.15407965183258057
Epoch 110 loss 1.3664256973803732 correct 50 Time Taken: 0.15250325202941895
Epoch 120 loss 1.5309462115001744 correct 50 Time Taken: 0.15642352104187013
Epoch 130 loss 0.843760424235691 correct 50 Time Taken: 0.2724982738494873
Epoch 140 loss 1.2906965316497514 correct 50 Time Taken: 0.29077045917510985
Epoch 150 loss 0.6348446534637202 correct 50 Time Taken: 0.32927398681640624
Epoch 160 loss 0.5122015862709333 correct 50 Time Taken: 0.3424734830856323
Epoch 170 loss 0.9275470104203024 correct 50 Time Taken: 0.24667294025421144
Epoch 180 loss 0.6541524387059358 correct 50 Time Taken: 0.15305893421173095
Epoch 190 loss 0.9617772017146343 correct 50 Time Taken: 0.15259602069854736
Epoch 200 loss 1.2412098262221907 correct 50 Time Taken: 0.15361194610595702
Epoch 210 loss 0.37017913835648847 correct 50 Time Taken: 0.15414214134216309
Epoch 220 loss 1.2163733980796845 correct 50 Time Taken: 0.15139491558074952
Epoch 230 loss 0.779285547788391 correct 50 Time Taken: 0.16652145385742187
Epoch 240 loss 0.7888676472587497 correct 50 Time Taken: 0.296448802947998
Epoch 250 loss 1.0642957769537045 correct 50 Time Taken: 0.301964545249939
Epoch 260 loss 0.9155899793138715 correct 50 Time Taken: 0.3458282709121704
Epoch 270 loss 0.5601529469250992 correct 50 Time Taken: 0.3542908191680908
Epoch 280 loss 0.9488792983779017 correct 50 Time Taken: 0.19799454212188722
Epoch 290 loss 0.8347085444914372 correct 50 Time Taken: 0.15214171409606933
Epoch 300 loss 0.7000723623108386 correct 50 Time Taken: 0.1527867794036865
Epoch 310 loss 0.5600001243655026 correct 50 Time Taken: 0.1532204866409302
Epoch 320 loss 0.8735732391491711 correct 50 Time Taken: 0.1538475275039673
Epoch 330 loss 0.3638758557272933 correct 50 Time Taken: 0.15517659187316896
Epoch 340 loss 0.05611130711005114 correct 50 Time Taken: 0.17992641925811767
Epoch 350 loss 0.4491666190297074 correct 50 Time Taken: 0.2736853361129761
Epoch 360 loss 0.5610487845022363 correct 50 Time Taken: 0.28088271617889404
Epoch 370 loss 0.24495520099417623 correct 50 Time Taken: 0.32541255950927733
Epoch 380 loss 0.6526486459823069 correct 50 Time Taken: 0.3325058937072754
Epoch 390 loss 0.04660024522915089 correct 50 Time Taken: 0.226621675491333
Epoch 400 loss 0.8623576173615317 correct 50 Time Taken: 0.15354812145233154
Epoch 410 loss 0.44879527845600653 correct 50 Time Taken: 0.15219337940216066
Epoch 420 loss 0.4750563996335171 correct 50 Time Taken: 0.15256903171539307
Epoch 430 loss 0.1308813722001444 correct 50 Time Taken: 0.16070692539215087
Epoch 440 loss 0.42016410332948106 correct 50 Time Taken: 0.15125200748443604
Epoch 450 loss 0.2897912100538334 correct 50 Time Taken: 0.16658830642700195
Epoch 460 loss 0.06846795175527613 correct 50 Time Taken: 0.2994158983230591
Epoch 470 loss 0.39762148699117533 correct 50 Time Taken: 0.29958310127258303
Epoch 480 loss 0.16295839333939077 correct 50 Time Taken: 0.3493041038513184
Epoch 490 loss 0.2020041453306346 correct 50 Time Taken: 0.35995633602142335
GPU:
Epoch 0 loss 7.974933467918762 correct 30 Time Taken: 4.95660138130188
Epoch 10 loss 4.157845765542286 correct 37 Time Taken: 2.4006747722625734
Epoch 20 loss 4.622342922586133 correct 39 Time Taken: 2.408205270767212
Epoch 30 loss 4.373049052859052 correct 42 Time Taken: 2.388383221626282
Epoch 40 loss 3.8886304530069618 correct 45 Time Taken: 2.3731048107147217
Epoch 50 loss 2.845774369329421 correct 46 Time Taken: 2.415011692047119
Epoch 60 loss 2.487651679395329 correct 45 Time Taken: 2.396859812736511
Epoch 70 loss 2.872602002027552 correct 48 Time Taken: 2.3883790731430055
Epoch 80 loss 2.2143704968819087 correct 47 Time Taken: 2.8453131437301638
Epoch 90 loss 2.329312527204623 correct 47 Time Taken: 2.448043942451477
Epoch 100 loss 1.3903733349442988 correct 48 Time Taken: 2.4247968196868896
Epoch 110 loss 1.1500483483440718 correct 47 Time Taken: 2.473524880409241
Epoch 120 loss 2.272556310802462 correct 47 Time Taken: 2.4836048603057863
Epoch 130 loss 1.0139381002811134 correct 47 Time Taken: 2.432215690612793
Epoch 140 loss 1.7080003550725582 correct 48 Time Taken: 2.38630006313324
Epoch 150 loss 1.3280698606881964 correct 48 Time Taken: 2.4152971267700196
Epoch 160 loss 1.9958774621332631 correct 50 Time Taken: 2.4038495063781737
Epoch 170 loss 2.4640720290628075 correct 43 Time Taken: 2.395455002784729
Epoch 180 loss 1.7848244155790032 correct 48 Time Taken: 2.383951950073242
Epoch 190 loss 1.5331280621232377 correct 47 Time Taken: 2.3923845529556274
Epoch 200 loss 0.5212988210510168 correct 48 Time Taken: 2.820062494277954
Epoch 210 loss 1.2753728653885563 correct 49 Time Taken: 2.3905029296875
Epoch 220 loss 1.3444037143318697 correct 50 Time Taken: 2.368544888496399
Epoch 230 loss 1.3579075264335605 correct 50 Time Taken: 2.4140047311782835
Epoch 240 loss 2.2620192163420456 correct 48 Time Taken: 2.3851658582687376
Epoch 250 loss 0.17973477528625223 correct 50 Time Taken: 2.384912109375
Epoch 260 loss 0.26311327944285007 correct 48 Time Taken: 2.4320645332336426
Epoch 270 loss 0.8235514590149599 correct 49 Time Taken: 2.3877299308776854
Epoch 280 loss 0.5217105445687623 correct 48 Time Taken: 2.448595952987671
Epoch 290 loss 0.4350463208854962 correct 50 Time Taken: 2.368028426170349
Epoch 300 loss 0.2542664346595303 correct 50 Time Taken: 2.4057321786880492
Epoch 310 loss 0.9045836871778955 correct 50 Time Taken: 2.8427385807037355
Epoch 320 loss 2.0258208988596804 correct 50 Time Taken: 2.396600079536438
Epoch 330 loss 0.8953808130778285 correct 48 Time Taken: 2.3784428596496583
Epoch 340 loss 0.31741680137729494 correct 49 Time Taken: 2.395302152633667
Epoch 350 loss 0.1458166113397411 correct 50 Time Taken: 2.40419864654541
Epoch 360 loss 2.3392822628769845 correct 44 Time Taken: 2.4298007249832154
Epoch 370 loss 0.08306883520653709 correct 48 Time Taken: 2.4051827907562258
Epoch 380 loss 0.8691262499578455 correct 50 Time Taken: 2.387833261489868
Epoch 390 loss 0.13294640692405013 correct 48 Time Taken: 2.3973992109298705
Epoch 400 loss 0.0570697437119138 correct 48 Time Taken: 2.3981209993362427
Epoch 410 loss 0.20483473612018913 correct 49 Time Taken: 2.3924572706222533
Epoch 420 loss 0.1656393319987139 correct 50 Time Taken: 2.3981674194335936
Epoch 430 loss 0.940429490942102 correct 50 Time Taken: 2.693231725692749
Epoch 440 loss 0.8115781514436765 correct 50 Time Taken: 2.3801382064819334
Epoch 450 loss 0.5432905996142335 correct 50 Time Taken: 2.392333507537842
Epoch 460 loss 0.194339321178185 correct 49 Time Taken: 2.3919334173202516
Epoch 470 loss 1.2165363414223147 correct 50 Time Taken: 2.401578187942505
Epoch 480 loss 0.06406111766263498 correct 49 Time Taken: 2.3757320642471313
Epoch 490 loss 0.31638205998319807 correct 49 Time Taken: 2.4001534223556518
Larger Model (200 hidden layers) GPU:
Epoch 0 loss 0.4489210494758966 correct 37 Time Taken: 3.8484206199645996
Epoch 10 loss 2.275113017380517 correct 40 Time Taken: 2.594332528114319
Epoch 20 loss 3.591103189078855 correct 43 Time Taken: 2.681138348579407
Epoch 30 loss 0.9724466719487529 correct 50 Time Taken: 3.4314425468444822
Epoch 40 loss 2.019698596074814 correct 46 Time Taken: 2.625036883354187
Epoch 50 loss 1.6457765482325162 correct 50 Time Taken: 2.5821751594543456
Epoch 60 loss 1.6306713096697942 correct 49 Time Taken: 2.5730797052383423
Epoch 70 loss 0.4579063898659655 correct 47 Time Taken: 2.5773000001907347
Epoch 80 loss 0.6659565923851475 correct 50 Time Taken: 2.5875149726867677
Epoch 90 loss 0.7385629836206701 correct 50 Time Taken: 2.728260350227356
Epoch 100 loss 1.0423026517832996 correct 50 Time Taken: 2.717244100570679
Epoch 110 loss 0.6141967585760385 correct 50 Time Taken: 2.749850559234619
Epoch 120 loss 0.08547160782198636 correct 50 Time Taken: 2.7413964748382567
Epoch 130 loss 0.6386580852871937 correct 50 Time Taken: 2.757072925567627
Epoch 140 loss 0.04398495846065897 correct 49 Time Taken: 2.6984042882919312
Epoch 150 loss 0.3210365094819372 correct 50 Time Taken: 2.5885738849639894
Epoch 160 loss 0.2753839311280433 correct 50 Time Taken: 2.6678750276565553
Epoch 170 loss 0.2551042425029594 correct 50 Time Taken: 2.7341165065765383
Epoch 180 loss 0.5112018035000716 correct 50 Time Taken: 2.75245943069458
Epoch 190 loss 0.022664385425972523 correct 50 Time Taken: 2.7497625589370727
Epoch 200 loss 0.0687981857128567 correct 50 Time Taken: 2.596474027633667
Epoch 210 loss 0.31050103760330305 correct 50 Time Taken: 2.574747157096863
Epoch 220 loss 0.020772672300640744 correct 50 Time Taken: 2.581787919998169
Epoch 230 loss 0.5015962340570872 correct 50 Time Taken: 2.5995918035507204
Epoch 240 loss 0.6209909832970502 correct 50 Time Taken: 2.836544418334961
Epoch 250 loss 0.47304675816153385 correct 50 Time Taken: 2.7062865257263184
Epoch 260 loss 0.5997899072726476 correct 50 Time Taken: 2.726625657081604
Epoch 270 loss 0.06760247516166765 correct 50 Time Taken: 2.7243634939193724
Epoch 280 loss 0.16457712680515235 correct 50 Time Taken: 2.660448122024536
Epoch 290 loss 0.16287981588692618 correct 50 Time Taken: 2.5895013332366945
Epoch 300 loss 0.16739161018761867 correct 50 Time Taken: 2.5799447536468505
Epoch 310 loss 0.2199551946713438 correct 50 Time Taken: 2.5808502674102782
Epoch 320 loss 0.35931063123932816 correct 50 Time Taken: 2.5936635971069335
Epoch 330 loss 0.22946894443268256 correct 50 Time Taken: 2.6753811597824098
Epoch 340 loss 0.1212375851645299 correct 50 Time Taken: 3.458565092086792
Epoch 350 loss 0.3994917091098038 correct 50 Time Taken: 2.7273213624954225
Epoch 360 loss 0.21160177318012816 correct 50 Time Taken: 2.572382950782776
Epoch 370 loss 0.008172202292508219 correct 50 Time Taken: 2.5750353813171385
Epoch 380 loss 0.04511659007603543 correct 50 Time Taken: 2.589924955368042
Epoch 390 loss 0.2958180896306731 correct 50 Time Taken: 2.5662872791290283
Epoch 400 loss 0.43871461695131103 correct 50 Time Taken: 2.6454338312149046
Epoch 410 loss 0.3728871753253495 correct 50 Time Taken: 2.723438549041748
Epoch 420 loss 0.0025841586994461863 correct 50 Time Taken: 2.7056568384170534
Epoch 430 loss 0.009844200077721778 correct 50 Time Taken: 2.7054975986480714
Epoch 440 loss 0.012140836700258681 correct 50 Time Taken: 2.7813111782073974
Epoch 450 loss 0.2842390786717613 correct 50 Time Taken: 2.665285849571228
Epoch 460 loss 0.04594095052934098 correct 50 Time Taken: 2.5820263385772706
Epoch 470 loss 0.005135485014862018 correct 50 Time Taken: 2.580795502662659
Epoch 480 loss 0.04407289997072914 correct 50 Time Taken: 2.5753418684005736
Epoch 490 loss 0.3192434517087379 correct 50 Time Taken: 2.6752658605575563

Simple:
CPU:
Epoch 0 loss 4.2212477746329755 correct 46 Time Taken: 15.288774013519287
Epoch 10 loss 1.8221468293938694 correct 49 Time Taken: 0.3475452423095703
Epoch 20 loss 0.8250616111211472 correct 49 Time Taken: 0.3432901859283447
Epoch 30 loss 0.9786608725725672 correct 49 Time Taken: 0.2655920505523682
Epoch 40 loss 0.9827949978339007 correct 50 Time Taken: 0.21975841522216796
Epoch 50 loss 0.5251314204065731 correct 49 Time Taken: 0.2959984064102173
Epoch 60 loss 0.3542142719453028 correct 50 Time Taken: 0.34127943515777587
Epoch 70 loss 0.9595496995230555 correct 50 Time Taken: 0.32543990612030027
Epoch 80 loss 0.8285409192612857 correct 50 Time Taken: 0.33707590103149415
Epoch 90 loss 0.36999359134755594 correct 50 Time Taken: 0.33950009346008303
Epoch 100 loss 0.17122708353707197 correct 50 Time Taken: 0.3456604480743408
Epoch 110 loss 0.32907332594193006 correct 50 Time Taken: 0.3428116083145142
Epoch 120 loss 0.22864597450490987 correct 50 Time Taken: 0.16041033267974852
Epoch 130 loss 0.07204700999000725 correct 50 Time Taken: 0.15235722064971924
Epoch 140 loss 0.1444492745788017 correct 50 Time Taken: 0.15227866172790527
Epoch 150 loss 0.21572368970485367 correct 50 Time Taken: 0.1545095205307007
Epoch 160 loss 0.0042955673185421 correct 50 Time Taken: 0.1517853021621704
Epoch 170 loss 0.28964628195291303 correct 50 Time Taken: 0.15250146389007568
Epoch 180 loss 0.013242480686687333 correct 50 Time Taken: 0.19068725109100343
Epoch 190 loss 0.026969815868010557 correct 50 Time Taken: 0.2834478378295898
Epoch 200 loss 0.14508329736964587 correct 50 Time Taken: 0.3368964433670044
Epoch 210 loss 0.08990984896674534 correct 50 Time Taken: 0.3568363904953003
Epoch 220 loss 0.13193027190795015 correct 50 Time Taken: 0.3310605525970459
Epoch 230 loss 0.08153638038182044 correct 50 Time Taken: 0.1522883415222168
Epoch 240 loss 0.00011987882977084834 correct 50 Time Taken: 0.15123205184936522
Epoch 250 loss 0.053836594778034236 correct 50 Time Taken: 0.15257635116577148
Epoch 260 loss 0.01402393604431533 correct 50 Time Taken: 0.15448732376098634
Epoch 270 loss 0.265806491879781 correct 50 Time Taken: 0.15440175533294678
Epoch 280 loss 0.1354140958977145 correct 50 Time Taken: 0.15148420333862306
Epoch 290 loss 0.0011284724241066232 correct 50 Time Taken: 0.20485491752624513
Epoch 300 loss 0.1927819593464231 correct 50 Time Taken: 0.2623048067092896
Epoch 310 loss 0.11087623602839489 correct 50 Time Taken: 0.3354297161102295
Epoch 320 loss 0.14666245510219772 correct 50 Time Taken: 0.3579885244369507
Epoch 330 loss 0.24904513245556192 correct 50 Time Taken: 0.346802806854248
Epoch 340 loss 0.1711698076497663 correct 50 Time Taken: 0.1560581922531128
Epoch 350 loss 0.0006929276804873305 correct 50 Time Taken: 0.15530531406402587
Epoch 360 loss 0.1353160802197228 correct 50 Time Taken: 0.1554657220840454
Epoch 370 loss 0.043999079944178066 correct 50 Time Taken: 0.15501725673675537
Epoch 380 loss 0.029823672981258747 correct 50 Time Taken: 0.1545037269592285
Epoch 390 loss 0.018302640163515686 correct 50 Time Taken: 0.1540318489074707
Epoch 400 loss 0.029585941957418156 correct 50 Time Taken: 0.22846605777740478
Epoch 410 loss 0.025377197278859613 correct 50 Time Taken: 0.28140809535980227
Epoch 420 loss 0.03458556992733351 correct 50 Time Taken: 0.34837889671325684
Epoch 430 loss 0.023547802947544903 correct 50 Time Taken: 0.36521809101104735
Epoch 440 loss 0.03309689445695744 correct 50 Time Taken: 0.2943428993225098
Epoch 450 loss 0.045267739924908144 correct 50 Time Taken: 0.15754354000091553
Epoch 460 loss 0.029277487160090324 correct 50 Time Taken: 0.15656213760375975
Epoch 470 loss 0.16058405684709554 correct 50 Time Taken: 0.15788187980651855
Epoch 480 loss 0.06927227799714275 correct 50 Time Taken: 0.15346043109893798
Epoch 490 loss 0.0827850786645643 correct 50 Time Taken: 0.1531909942626953
GPU:
Epoch 0 loss 4.716978612328049 correct 45 Time Taken: 7.50085711479187
Epoch 10 loss 1.814733498927817 correct 49 Time Taken: 2.4152930974960327
Epoch 20 loss 0.7918220415599531 correct 49 Time Taken: 2.3956226825714113
Epoch 30 loss 0.9512973235538765 correct 47 Time Taken: 2.3821316003799438
Epoch 40 loss 0.5004428612810045 correct 50 Time Taken: 2.3968220233917235
Epoch 50 loss 1.3833468072629338 correct 46 Time Taken: 2.8126137495040893
Epoch 60 loss 0.6668770807106678 correct 46 Time Taken: 2.3911020517349244
Epoch 70 loss 0.2717526815161546 correct 47 Time Taken: 2.3903494596481325
Epoch 80 loss 0.16940432833254526 correct 50 Time Taken: 2.3944819450378416
Epoch 90 loss 1.2180129518606342 correct 48 Time Taken: 2.3894602775573732
Epoch 100 loss 1.2875674896045137 correct 50 Time Taken: 2.393473672866821
Epoch 110 loss 0.12367052166084547 correct 48 Time Taken: 2.399209761619568
Epoch 120 loss 0.016307335440802775 correct 50 Time Taken: 2.394455146789551
Epoch 130 loss 1.2213799738005466 correct 50 Time Taken: 2.3836445093154905
Epoch 140 loss 0.7350533747799705 correct 50 Time Taken: 2.3736191034317016
Epoch 150 loss 0.7874597813978136 correct 50 Time Taken: 2.388503408432007
Epoch 160 loss 0.5652531727708158 correct 50 Time Taken: 2.6036115884780884
Epoch 170 loss 0.5614965054428378 correct 50 Time Taken: 2.4040220975875854
Epoch 180 loss 0.006006496455524153 correct 48 Time Taken: 2.3960474014282225
Epoch 190 loss 1.0084869789260915 correct 50 Time Taken: 2.3688135862350466
Epoch 200 loss 0.2873035630782675 correct 50 Time Taken: 2.3730724573135378
Epoch 210 loss 0.7111208617762994 correct 50 Time Taken: 2.381368899345398
Epoch 220 loss 0.036102481429779155 correct 50 Time Taken: 2.3812692403793334
Epoch 230 loss 0.32388175263124547 correct 50 Time Taken: 2.385881471633911
Epoch 240 loss 0.7306554180174711 correct 50 Time Taken: 2.3836665630340574
Epoch 250 loss 1.0439610173341283 correct 48 Time Taken: 2.40067982673645
Epoch 260 loss 0.0005857401468172724 correct 50 Time Taken: 2.3860083341598513
Epoch 270 loss 0.3053454350813891 correct 50 Time Taken: 2.61095712184906
Epoch 280 loss 0.5740911473972711 correct 50 Time Taken: 2.382842254638672
Epoch 290 loss 0.16495038344750515 correct 50 Time Taken: 2.3729328870773316
Epoch 300 loss 0.005202959864817025 correct 50 Time Taken: 2.382869243621826
Epoch 310 loss 0.28823211889755407 correct 50 Time Taken: 2.3857847452163696
Epoch 320 loss 0.06703536957694 correct 50 Time Taken: 2.3883440017700197
Epoch 330 loss 0.03475970948394657 correct 50 Time Taken: 2.3966697216033936
Epoch 340 loss 0.13092489700282708 correct 50 Time Taken: 2.3839999198913575
Epoch 350 loss 0.03447520536021406 correct 50 Time Taken: 2.393682360649109
Epoch 360 loss 0.007454591038705667 correct 50 Time Taken: 2.382278871536255
Epoch 370 loss 0.4166095663481028 correct 50 Time Taken: 2.3833876609802247
Epoch 380 loss 0.058933425788729364 correct 50 Time Taken: 2.7813875675201416
Epoch 390 loss 0.0020813827286587957 correct 50 Time Taken: 2.3891969442367555
Epoch 400 loss 0.5810069235758504 correct 50 Time Taken: 2.40742666721344
Epoch 410 loss 0.6379838597769767 correct 50 Time Taken: 2.400341844558716
Epoch 420 loss 0.34182094898070503 correct 50 Time Taken: 2.391498589515686
Epoch 430 loss 0.0003713456707043265 correct 50 Time Taken: 2.3692617654800414
Epoch 440 loss 0.053229981661660354 correct 50 Time Taken: 2.3853331804275513
Epoch 450 loss 0.0835218444952649 correct 50 Time Taken: 2.4189989805221557
Epoch 460 loss 0.33639441423232624 correct 50 Time Taken: 2.378022861480713
Epoch 470 loss 0.41139944240566567 correct 50 Time Taken: 2.3855549812316896
Epoch 480 loss 0.3881566189212681 correct 50 Time Taken: 2.3827425956726076
Epoch 490 loss 0.0021618897508002685 correct 50 Time Taken: 3.023139762878418
GPU Large Model (300 Units):
Epoch 0 loss 40.23118160034231 correct 43 Time Taken: 6.5791826248168945
Epoch 10 loss 0.1563755715130418 correct 48 Time Taken: 2.755997371673584
Epoch 20 loss 0.34580198008600666 correct 49 Time Taken: 2.6776218891143797
Epoch 30 loss 0.2810935773016736 correct 49 Time Taken: 2.6011088132858275
Epoch 40 loss 0.3657471215682274 correct 50 Time Taken: 2.9330013036727904
Epoch 50 loss 1.2228074113011358 correct 50 Time Taken: 2.681252098083496
Epoch 60 loss 0.0019131696512976295 correct 50 Time Taken: 2.759407711029053
Epoch 70 loss 1.4217331136716944 correct 50 Time Taken: 2.7918960809707642
Epoch 80 loss 1.5909042026280151 correct 48 Time Taken: 2.75585823059082
Epoch 90 loss 0.3628046467101651 correct 49 Time Taken: 2.614917778968811
Epoch 100 loss 0.03778332297037318 correct 48 Time Taken: 2.573711371421814
Epoch 110 loss 0.0906851792401003 correct 49 Time Taken: 2.6065600395202635
Epoch 120 loss 0.058493276533104335 correct 49 Time Taken: 2.6088136196136475
Epoch 130 loss 1.1226337423787427 correct 49 Time Taken: 3.1836458921432493
Epoch 140 loss 0.021854204331888952 correct 49 Time Taken: 2.7993300437927244
Epoch 150 loss 0.007739881044059232 correct 48 Time Taken: 2.767517924308777
Epoch 160 loss 0.2189847306316983 correct 48 Time Taken: 2.6080281019210814
Epoch 170 loss 0.060953657083563734 correct 48 Time Taken: 2.6052940845489503
Epoch 180 loss 1.1795039481373277 correct 49 Time Taken: 2.5979082345962525
Epoch 190 loss 0.020656902207926847 correct 48 Time Taken: 2.5940889358520507
Epoch 200 loss 0.8301127847254653 correct 49 Time Taken: 2.688437271118164
Epoch 210 loss 1.1439730911626782 correct 49 Time Taken: 2.737553572654724
Epoch 220 loss 1.2973491920928668 correct 49 Time Taken: 2.7819342374801637
Epoch 230 loss 0.11990156369180513 correct 49 Time Taken: 3.098752760887146
Epoch 240 loss 0.33775024996323394 correct 49 Time Taken: 2.605518627166748
Epoch 250 loss 0.10456351669752058 correct 49 Time Taken: 2.6342141151428224
Epoch 260 loss 1.283575006422978 correct 48 Time Taken: 2.6625261306762695
Epoch 270 loss 0.2822760266629695 correct 50 Time Taken: 2.7275019884109497
Epoch 280 loss 1.126423161205734 correct 49 Time Taken: 2.7699598550796507
Epoch 290 loss 0.011000435763175945 correct 49 Time Taken: 2.8296390771865845
Epoch 300 loss 0.17119145813849224 correct 49 Time Taken: 2.663252663612366
Epoch 310 loss 0.0820216480060791 correct 49 Time Taken: 2.644380521774292
Epoch 320 loss 0.12077495963937132 correct 49 Time Taken: 2.8350855112075806
Epoch 330 loss 1.9015893585670987 correct 48 Time Taken: 2.6487753868103026
Epoch 340 loss 0.43655568419264246 correct 49 Time Taken: 2.755573034286499
Epoch 350 loss 0.08573714758743009 correct 49 Time Taken: 2.8119113445281982
Epoch 360 loss 0.06535042789444016 correct 49 Time Taken: 2.780400776863098
Epoch 370 loss 0.008259114306145315 correct 49 Time Taken: 2.63266978263855
Epoch 380 loss 0.017227722305335515 correct 50 Time Taken: 2.5996954917907713
Epoch 390 loss 0.946308392443405 correct 49 Time Taken: 2.600300693511963
Epoch 400 loss 0.0616393911477782 correct 49 Time Taken: 2.6008710622787476
Epoch 410 loss 0.14174990203158 correct 49 Time Taken: 3.351065754890442
Epoch 420 loss 1.2328105041299455 correct 50 Time Taken: 2.7929932832717896
Epoch 430 loss 0.003936594298305188 correct 49 Time Taken: 2.6635474443435667
Epoch 440 loss 0.2335954414675836 correct 49 Time Taken: 2.6139903545379637
Epoch 450 loss 1.1626559588271272 correct 48 Time Taken: 2.593301296234131
Epoch 460 loss 1.277826473966085 correct 49 Time Taken: 2.6225021839141847
Epoch 470 loss 0.002424534769220593 correct 49 Time Taken: 2.6408914804458616
Epoch 480 loss 0.030540169356002185 correct 48 Time Taken: 2.7077836275100706
Epoch 490 loss 0.01847321920746436 correct 49 Time Taken: 2.7829140424728394
XOR:
CPU:
poch 0 loss 5.627348600631587 correct 29 Time Taken: 17.04417896270752
Epoch 10 loss 5.778817120100012 correct 46 Time Taken: 0.25033957958221437
Epoch 20 loss 4.110606199034677 correct 46 Time Taken: 0.2686979532241821
Epoch 30 loss 3.1154336503633386 correct 46 Time Taken: 0.34284875392913816
Epoch 40 loss 3.3681738733096585 correct 43 Time Taken: 0.34992907047271726
Epoch 50 loss 3.6185421452782536 correct 46 Time Taken: 0.2814438581466675
Epoch 60 loss 3.792304259223147 correct 43 Time Taken: 0.15625932216644287
Epoch 70 loss 3.9109435411614464 correct 44 Time Taken: 0.15414190292358398
Epoch 80 loss 5.477462496919388 correct 43 Time Taken: 0.15661196708679198
Epoch 90 loss 3.1075565894066095 correct 46 Time Taken: 0.16057119369506836
Epoch 100 loss 1.481704971294005 correct 47 Time Taken: 0.1558593988418579
Epoch 110 loss 0.5105031998133447 correct 47 Time Taken: 0.15388569831848145
Epoch 120 loss 1.6167288199808714 correct 45 Time Taken: 0.2682607889175415
Epoch 130 loss 0.9552914167098697 correct 47 Time Taken: 0.2927492380142212
Epoch 140 loss 1.8928736000590218 correct 47 Time Taken: 0.3511807441711426
Epoch 150 loss 3.5862339173686073 correct 46 Time Taken: 0.33341567516326903
Epoch 160 loss 0.2923054934280346 correct 45 Time Taken: 0.24639790058135985
Epoch 170 loss 1.952387533152713 correct 47 Time Taken: 0.15253927707672119
Epoch 180 loss 0.37301768154266485 correct 47 Time Taken: 0.15427854061126708
Epoch 190 loss 2.2165197961951097 correct 48 Time Taken: 0.15442683696746826
Epoch 200 loss 3.1375423877478275 correct 48 Time Taken: 0.1532686471939087
Epoch 210 loss 0.5387950211903616 correct 47 Time Taken: 0.15202245712280274
Epoch 220 loss 2.7424462763543174 correct 49 Time Taken: 0.1564728021621704
Epoch 230 loss 1.371945015800687 correct 48 Time Taken: 0.29125807285308836
Epoch 240 loss 1.9544714909770053 correct 48 Time Taken: 0.307720422744751
Epoch 250 loss 0.6918919787226557 correct 49 Time Taken: 0.34211976528167726
Epoch 260 loss 1.8854174124764433 correct 49 Time Taken: 0.33567850589752196
Epoch 270 loss 3.301598024320644 correct 46 Time Taken: 0.2085493326187134
Epoch 280 loss 0.8609962895638285 correct 48 Time Taken: 0.1537792682647705
Epoch 290 loss 0.5769154828117008 correct 49 Time Taken: 0.15366556644439697
Epoch 300 loss 0.6428156899703734 correct 49 Time Taken: 0.15232436656951903
Epoch 310 loss 0.19088247150181092 correct 49 Time Taken: 0.1513209104537964
Epoch 320 loss 1.2224623028600643 correct 49 Time Taken: 0.15449097156524658
Epoch 330 loss 0.6595495201982402 correct 49 Time Taken: 0.19262514114379883
Epoch 340 loss 1.078612102857358 correct 49 Time Taken: 0.29337196350097655
Epoch 350 loss 0.7120772058570649 correct 49 Time Taken: 0.3268823862075806
Epoch 360 loss 0.4953866107411003 correct 49 Time Taken: 0.3501795530319214
Epoch 370 loss 0.5822614415546515 correct 49 Time Taken: 0.3422394752502441
Epoch 380 loss 0.25377175325436047 correct 48 Time Taken: 0.272051739692688
Epoch 390 loss 0.40916007675258215 correct 50 Time Taken: 0.3452150344848633
Epoch 400 loss 0.9597077572045081 correct 49 Time Taken: 0.34448995590209963
Epoch 410 loss 0.6188296009018422 correct 49 Time Taken: 0.2977156400680542
Epoch 420 loss 1.8038176739118945 correct 50 Time Taken: 0.25856637954711914
Epoch 430 loss 2.1268787517150414 correct 49 Time Taken: 0.34245264530181885
Epoch 440 loss 0.8042131529494091 correct 49 Time Taken: 0.3454874277114868
Epoch 450 loss 1.9022196312824797 correct 49 Time Taken: 0.3247237682342529
Epoch 460 loss 0.2117989964457506 correct 49 Time Taken: 0.15241606235504152
Epoch 470 loss 0.6556118660678695 correct 49 Time Taken: 0.15171239376068116
Epoch 480 loss 1.3580751198334235 correct 49 Time Taken: 0.1529384136199951
Epoch 490 loss 0.2745716944335615 correct 50 Time Taken: 0.15306756496429444
GPU:
Epoch 0 loss 6.910452779679895 correct 36 Time Taken: 4.67685866355896
Epoch 10 loss 4.606319366910728 correct 46 Time Taken: 2.4141568183898925
Epoch 20 loss 2.123775660646786 correct 47 Time Taken: 2.3769452810287475
Epoch 30 loss 2.024177649540328 correct 45 Time Taken: 2.359957456588745
Epoch 40 loss 4.090510383611869 correct 47 Time Taken: 2.338688683509827
Epoch 50 loss 3.8186372260747987 correct 49 Time Taken: 2.3487894535064697
Epoch 60 loss 2.3695906867687793 correct 49 Time Taken: 2.36821653842926
Epoch 70 loss 2.3414321297405434 correct 49 Time Taken: 2.352563261985779
Epoch 80 loss 1.9500521845357086 correct 50 Time Taken: 2.342136287689209
Epoch 90 loss 1.809925431820252 correct 47 Time Taken: 2.353811526298523
Epoch 100 loss 1.112650527968676 correct 48 Time Taken: 3.0221399784088137
Epoch 110 loss 1.476399434601742 correct 49 Time Taken: 2.3489428758621216
Epoch 120 loss 0.8708856733323618 correct 49 Time Taken: 2.3781171560287477
Epoch 130 loss 1.1591958561044515 correct 50 Time Taken: 2.365823578834534
Epoch 140 loss 1.3292159072376808 correct 50 Time Taken: 2.346937966346741
Epoch 150 loss 0.7569735052510838 correct 48 Time Taken: 2.340663456916809
Epoch 160 loss 0.4093819683395943 correct 50 Time Taken: 2.367245411872864
Epoch 170 loss 0.6617537140796286 correct 50 Time Taken: 2.3641302585601807
Epoch 180 loss 0.8276471236666919 correct 49 Time Taken: 2.342658257484436
Epoch 190 loss 0.7703811666374156 correct 50 Time Taken: 2.3431012630462646
Epoch 200 loss 0.721660680380381 correct 50 Time Taken: 2.367547702789307
Epoch 210 loss 1.8190518787035932 correct 49 Time Taken: 2.347751808166504
Epoch 220 loss 1.185913675248436 correct 50 Time Taken: 2.3307469367980955
Epoch 230 loss 0.9174430794545935 correct 50 Time Taken: 2.36511390209198
Epoch 240 loss 1.7093213867381263 correct 50 Time Taken: 2.554989385604858
Epoch 250 loss 1.2610394357680157 correct 50 Time Taken: 2.3881708860397337
Epoch 260 loss 0.914333976777729 correct 50 Time Taken: 2.3654393672943117
Epoch 270 loss 1.3489280266138708 correct 49 Time Taken: 2.3604233503341674
Epoch 280 loss 0.5761787097659158 correct 50 Time Taken: 2.3458689212799073
Epoch 290 loss 0.7062320769840433 correct 50 Time Taken: 2.3361722946166994
Epoch 300 loss 0.8626456750004612 correct 50 Time Taken: 2.3667107105255125
Epoch 310 loss 0.3078532397857112 correct 50 Time Taken: 2.3468031167984007
Epoch 320 loss 0.7529602289120384 correct 50 Time Taken: 2.3525522232055662
Epoch 330 loss 1.0165711115081046 correct 50 Time Taken: 2.3590953588485717
Epoch 340 loss 0.8196851051839936 correct 50 Time Taken: 2.3371116161346435
Epoch 350 loss 1.335427134220582 correct 50 Time Taken: 2.3646653175354
Epoch 360 loss 0.8652235378438523 correct 50 Time Taken: 2.4032230138778687
Epoch 370 loss 0.6052634788660334 correct 49 Time Taken: 2.3557692527770997
Epoch 380 loss 0.8691616854074947 correct 49 Time Taken: 2.3451597690582275
Epoch 390 loss 0.5203250877727806 correct 50 Time Taken: 3.072075629234314
Epoch 400 loss 0.9991080440185997 correct 50 Time Taken: 2.3853713035583497
Epoch 410 loss 0.2727130373744487 correct 49 Time Taken: 2.356326460838318
Epoch 420 loss 0.044249022611772805 correct 50 Time Taken: 2.3584429264068603
Epoch 430 loss 0.14734551041346844 correct 50 Time Taken: 2.3744116544723513
Epoch 440 loss 0.7603018859306504 correct 50 Time Taken: 2.36371853351593
Epoch 450 loss 1.9998412122516471 correct 48 Time Taken: 2.36928927898407
Epoch 460 loss 0.44767266271992795 correct 49 Time Taken: 2.353863978385925
Epoch 470 loss 0.12983846614787506 correct 50 Time Taken: 2.35291702747345
Epoch 480 loss 0.024310733914492285 correct 50 Time Taken: 2.3738092184066772
Epoch 490 loss 0.34823606943242225 correct 50 Time Taken: 2.418120336532593
GPU Large Model (300 models):
Epoch 0 loss 10.814947722382461 correct 20 Time Taken: 5.085723876953125
Epoch 10 loss 2.23368340958008 correct 44 Time Taken: 2.6315419912338256
Epoch 20 loss 2.2559397222082573 correct 41 Time Taken: 2.7873212814331056
Epoch 30 loss 1.3372157098588866 correct 45 Time Taken: 2.755616569519043
Epoch 40 loss 2.922448439198838 correct 45 Time Taken: 2.56074800491333
Epoch 50 loss 2.7707922716083324 correct 40 Time Taken: 2.5804891109466555
Epoch 60 loss 3.2751754077104933 correct 43 Time Taken: 2.6305499792099
Epoch 70 loss 2.956717212806547 correct 43 Time Taken: 2.80561306476593
Epoch 80 loss 0.32583674799767054 correct 48 Time Taken: 2.817000079154968
Epoch 90 loss 0.6715645381179516 correct 49 Time Taken: 2.603768539428711
Epoch 100 loss 0.37280796320390913 correct 49 Time Taken: 2.571745491027832
Epoch 110 loss 1.0604691310322174 correct 50 Time Taken: 2.5915724277496337
Epoch 120 loss 0.953071416357212 correct 50 Time Taken: 2.7852306604385375
Epoch 130 loss 0.25922657670463517 correct 50 Time Taken: 2.833358716964722
Epoch 140 loss 0.9671825739185547 correct 50 Time Taken: 2.645137667655945
Epoch 150 loss 0.5218153309229685 correct 50 Time Taken: 2.7896934270858766
Epoch 160 loss 0.28940351702045425 correct 50 Time Taken: 2.6654520273208617
Epoch 170 loss 0.5378803488483461 correct 50 Time Taken: 2.8300913095474245
Epoch 180 loss 0.8428285815144201 correct 50 Time Taken: 2.793291687965393
Epoch 190 loss 0.20854962670244323 correct 50 Time Taken: 2.5906220436096192
Epoch 200 loss 0.6909211258708389 correct 50 Time Taken: 2.5971397161483765
Epoch 210 loss 0.20221676902716193 correct 50 Time Taken: 2.646300768852234
Epoch 220 loss 0.3757937418421413 correct 50 Time Taken: 2.8184089422225953
Epoch 230 loss 0.25957619606235244 correct 50 Time Taken: 2.7689508199691772
Epoch 240 loss 0.6249872282313792 correct 50 Time Taken: 2.6016371726989744
Epoch 250 loss 0.4073510727737241 correct 50 Time Taken: 2.58032603263855
Epoch 260 loss 0.5057069934700752 correct 50 Time Taken: 2.637078046798706
Epoch 270 loss 0.1514374991412736 correct 50 Time Taken: 2.851837420463562
Epoch 280 loss 0.7830979322400198 correct 50 Time Taken: 2.810435962677002
Epoch 290 loss 0.23605371904493871 correct 50 Time Taken: 2.6026238441467284
Epoch 300 loss 0.17405179211587163 correct 50 Time Taken: 2.57781982421875
Epoch 310 loss 0.3334732443918438 correct 50 Time Taken: 2.6314017534255982
Epoch 320 loss 0.24057456812961706 correct 50 Time Taken: 2.806999754905701
Epoch 330 loss 0.2361994962133167 correct 50 Time Taken: 2.831048083305359
Epoch 340 loss 0.23219538353026145 correct 50 Time Taken: 2.59058518409729
Epoch 350 loss 0.2565529385075897 correct 50 Time Taken: 2.5743836164474487
Epoch 360 loss 0.2737914448409245 correct 50 Time Taken: 2.631417679786682
Epoch 370 loss 0.06864752561214434 correct 50 Time Taken: 2.8206281661987305
Epoch 380 loss 0.3420080191271091 correct 50 Time Taken: 2.8508232831954956
Epoch 390 loss 0.4289410728360797 correct 50 Time Taken: 2.6096630573272703
Epoch 400 loss 0.21052587913281243 correct 50 Time Taken: 2.6069284200668337
Epoch 410 loss 0.07806745509884915 correct 50 Time Taken: 3.2404919385910036
Epoch 420 loss 0.25534710707897607 correct 50 Time Taken: 2.8297491550445555
Epoch 430 loss 0.24547214838431197 correct 50 Time Taken: 2.596785044670105
Epoch 440 loss 0.06387066781035969 correct 50 Time Taken: 2.574781584739685
Epoch 450 loss 0.14752701732340767 correct 50 Time Taken: 2.5855604648590087
Epoch 460 loss 0.1474778930858269 correct 50 Time Taken: 2.8181107521057127
Epoch 470 loss 0.06292358554177681 correct 50 Time Taken: 2.830213713645935
Epoch 480 loss 0.1474126318724354 correct 50 Time Taken: 2.6057613611221315
Epoch 490 loss 0.09360997615800118 correct 50 Time Taken: 2.607413125038147
