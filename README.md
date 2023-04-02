# Label-Assisted-Unpaired-Image-to-Image-Translation

This project describe the labeled-assisted unpaired image-to-image translation framework. The overall architetcire compromises 4 shared weight generator, 4 discriminators, and 2 pre-trained segmentation model.

## Model Architecture Overview

### Generators
1. source domain to target domain generator
2. target domain to source domain generator
3. source domain to target label domain generator
4. target domain to source label domain generator

### Discriminators
1. Source domain discriminator
2. Target domain discriminator
3. Partial Source domain discriminator
4. Partial Target domain discriminator

### Segmentor
1. Source domain segmentor
2. Target domain segmentor

## Loss function overview
1. Source domain GAN loss
2. Target domain GAN loss
3. Partial source domain GAN loss
4. Partial target domain GAN loss
5. Source domain cycle-consistency loss
6. Target domain cycle-consistency loss
7. Source label domain cycle-consistency loss
8. Target label domain cycle-consistency loss
9. Source domain supervised loss
10. Target domain supervised loss

## Dataset format

You should check out the size of A domain images are the same with B domain image.

```
./[your own path]/dataset
----/trainA   % A domain training set image file [png]
----/trainB   % B domain training set image file [png]
----/testA    % A domain testing set image file [png]
----/testB    % B domain testing set image file [png]
```

## Model Training

```
python train.py --dataroot ./[your own path]/dataset --name [your model name] --model cycle_gan --A_domain_segmentor [your pre-trained A segmentor] --B_domain_segmentor [your pre-trained B segmentor] --noise1 'YES' --noise2 'YES' --noise3 'YES' --noise4 'YES' 
```

## Model Testing
```
python test.py --dataroot ./[your own path]/dataset --name [your model name] --model cycle_gan --A_domain_segmentor [your pre-trained A segmentor] --B_domain_segmentor [your pre-trained B segmentor] --noise1 'YES' --noise2 'YES' --noise3 'YES' --noise4 'YES' 
```



