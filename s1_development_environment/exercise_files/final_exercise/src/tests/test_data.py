import sys
sys.path.insert(1, '../models/')
import train_model

def test_data():
    dataset_train, dataset_test = train_model.mnist()

    assert len(dataset_train) == 25000 
    assert len(dataset_test) == 5000

    for images, labels in dataset_train:
        assert images.shape == (1, 28, 28)
        assert labels.shape[-1] == 1

    for images, labels in dataset_test:
        assert images.shape == (1, 28, 28)
        assert labels.shape[-1] == 1