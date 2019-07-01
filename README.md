# train self wGAN with label in Pytorch 

## file overview:
wgan_gp.py(model structure)  
mydataset.py(image and label dataset with onehot label)  
mydataset2.py(image and label dataset with constant label)  
wgan_gp_train.py(train model and print generator result in interval)  
w_printpic.py(test result by label)

## model structure:
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/gmodel.png)
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/dmodel.png)

## how to train model:  
python3.6 wgan_gp_train.py [parameter]  
parameter list:  
--n_epochs  
--batch_size  
--lr  
--b1  
--b2  
--n_cpu  
--latent_dim  
--n_classes  
--img_size  
--channels  
--sample_interval  
--dataroot  
--labelroot  
--testlabel  
--n_row  

## result image for test labels in trained generator:  
python3.6 w_printpic.py [parameter]  
parameter list:  
--latent_dim  
--n_classes  
--img_size  
--channels  
--testlabel  
--outputdir  
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/epoch2.png)
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/epoch7.png)
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/epoch12.png)
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/epoch17.png)
![image](https://github.com/TaiChunYen/adl_HW4/blob/master/picture/epoch24.png)


## reference:
https://github.com/jalola/improved-wgan-pytorch  
https://google.github.io/cartoonset/download.html


