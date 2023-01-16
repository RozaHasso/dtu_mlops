import sys
sys.path.insert(1, '../models/')
import model
import torch

def test_model():
    myModel = model.MyAwesomeModel()
    img = torch.randn(1, 28, 28)
    output = myModel.forward(img)

    assert output.shape == (1,)
    assert output in range(10)