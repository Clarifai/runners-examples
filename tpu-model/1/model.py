import torch
import torch.nn as nn
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("torch_xla not found. This model requires a TPU environment.")
    torch_xla = None
    xm = None

from typing import List
from clarifai.runners.models.model_class import ModelClass

class SimpleTPUModel(nn.Module):
    def __init__(self):
        super(SimpleTPUModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class MyTPUModel(ModelClass):
  """
  This is a model that runs on TPUs.
  """

  def load_model(self):
    """
    Load the model and move it to the TPU device.
    """
    if xm:
        self.device = xm.xla_device()
        self.model = SimpleTPUModel()
        self.model.to(self.device)
        self.model.eval()
    else:
        self.model = None
        print("Skipping model loading due to missing torch_xla.")


  @ModelClass.method
  def predict(self, input_data: List[float]) -> List[float]:
    """
    Run inference on the model. The input should be a list of 10 floats.
    """
    if not self.model:
        return [0.0]

    input_tensor = torch.tensor(input_data).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(self.device)
    with torch.no_grad():
        output = self.model(input_tensor)
    return output.cpu().flatten().tolist()

def test_predict() -> None:
    """Test the predict method of MyTPUModel by printing its output."""
    if not xm:
        print("Cannot run test_predict without torch_xla and a TPU device.")
        return

    model = MyTPUModel()
    model.load_model()
    print("Testing predict method:")
    # Create a dummy input list of 10 floats
    input_data = [1.0] * 10
    output = model.predict(input_data)
    print(output)

if __name__ == "__main__":
    print("TPU model code created. To run the test, a TPU device and torch_xla are required.")
    # test_predict()
