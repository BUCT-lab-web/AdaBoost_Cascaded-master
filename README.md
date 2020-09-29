`the dataset:`</br>
Just put the data into the folder of 'datasets/class10/test' and 'datasets/class10/train',then the classes as:</br> 
1_new Az_2S1</br> 
2_new BMP2</br> 
3_new BRDM_2</br> 
4_new BTR_60</br> 
5_new BTR70</br> 
6_new D7</br> 
7_new T62</br> 
8_new T72</br> 
9_new ZIL131</br> 
10_new ZSU_23_4</br> 

`the models: `</br>
```python
models.py #our CNN models
othermodels_all.py #the method we used to compare the acc classified by the features extracted by CNN
othermodels.py #the method we used
A_ConvNets.py #the model of A_convnet
models_concate.py # Concatenate different layers by cnn
```
`the training: `</br>
```python
ours_method.py #our method
ours_method_onelayer.py #try different layer by our method
Aconvnet_compare.py #Aconvnet method
all_method.py #try other ensemble method
```
`the testing: `</br>
```python
test1.py #used all_method model
test2.py #uesd ours_method_onelayer model
test3.py #used models_concate model
test4.py #used ours_method model
```
