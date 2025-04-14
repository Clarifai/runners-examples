
import os
import sys

sys.path.append(os.path.dirname(__file__))
from openai_client_wrapper import OpenAIWrapper
#################
from typing import Iterator


from clarifai.runners.models.model_class import ModelClass
from clarifai_grpc.grpc.api import service_pb2
from openai import OpenAI

# Set your OpenAI API key here
API_KEY = 'XAI-API-KEY'


class Grok(ModelClass):
  """A custom runner that wraps the Openai GPT-4 model and generates text using it.
  """

  def load_model(self):
    """Load the model here."""
    # Create client
    self.client = OpenAIWrapper(
      client=OpenAI(api_key=API_KEY,base_url="https://api.x.ai/v1"),
      modalities=["image", "audio", "video"],
      model = "grok-3-latest",

    )
    # log that system is ready
    print("Grok model loaded successfully!")

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    return self.client.predict(request)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method generates stream of outputs for the given inputs in the request."""
    for each in self.client.generate(request):
      yield each

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("")
