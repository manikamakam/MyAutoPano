# MyAutoPano
This project aimed at stitching two or more images in order to create one seamless panorama image. This task requires homography estimation between two images. We calculate the homography between two images using the classical vision approach and deep learning approach (both supervised and unsupervised). 

## Phase 1

In phase 1, we implement the creation of a seamless panorama using tradiitonal geometric vision approach.

To run the code, run the following command from Phase1 folder:

```
python Wrapper.py
```

## Phase 2

In this phase, we estimate homography using deep learning approach. We implemented supervised and unsupervised models.

## Training the network

To train the supervised model, change the 'ModelType' argument to 'Sup' in the code and run the following command:

```
python Train.py
```

For training theunsupervised model, 'ModelType' should be 'Unsup'.


## Testing a network

To test the supervised model, run the command:

```
python test_sup.py
```

To test the unsupervised model, run the command:

```
python test_unsup.py
```
## Estimation of homography

To compare the estimated homographies with ground truth, run the following command:

```
pyhton Wrapper.py
```

## Results

<p align="center">
	<img src="https://github.com/p-akanksha/CMSC733/blob/master/Results/output1.jpg" width="400">
</p>

<p align="center">
	<img src="https://github.com/p-akanksha/CMSC733/blob/master/Results/output2.jpg" width="600">
</p>

## Author

1. Akanksha Patel
2. Sri Manika Makam
