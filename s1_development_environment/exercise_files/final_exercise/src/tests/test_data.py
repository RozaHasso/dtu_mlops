import sys
sys.path.insert(1, '../models/')
import train_model

'''Testing the length of the dataset'''
def test_length():
    dataset_train, dataset_test = train_model.mnist()

    assert len(dataset_train) == 25000
    assert len(dataset_test) == 5000

'''Testing the shape of each datapoint'''
def test_shape():
    dataset_train, dataset_test = train_model.mnist()

    _check_shape(dataset_train)
    _check_shape(dataset_test)

def _check_shape(data):
    for images, labels in data:
        assert images.shape == (1, 28, 28)
        assert labels.shape == (1,)
        assert labels in range(10)

'''Testing the scale of each datapoint'''
def test_scaling():
    dataset_train, dataset_test = train_model.mnist()

    _check_scaling(dataset_train)
    _check_scaling(dataset_test)

def _check_scaling(data):
    for images, labels in data:
        assert images.max() <= 1 
        assert labels.max() <= 1

        assert images.min() >= -1
        assert labels.min() >= -1


