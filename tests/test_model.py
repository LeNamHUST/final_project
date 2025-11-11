from src.models.resnet_model import RetinalResnetModel


model = RetinalResnetModel(num_classes=7)
def test_model():
    image = 
    with torch.no_grad():
        y = model(image)
    assert y.shape == (1, 7), "output error shape"