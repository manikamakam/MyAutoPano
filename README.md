## Author

Akanksha Patel
Sri Manika Makam


## Phase 1

In phase 1, we implement the creation of a seamless panorama using tradiitonal geometric vision approach. Outputs are shown in the report for Train, Validation and Test sets.

Run the following command:

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

The ouputs are shown in the report.

