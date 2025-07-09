# Clarifai Local Runner Setup

This folder contains examples that demonstrate how to create and set up Clarifai local runners for custom model development.

## Prerequisites

Before setting up this repository, ensure you have:
- Python 3.11+ installed
- pip package manager
- Git
- A Clarifai account

### Using Clarifai CLI for Local Testing

The Clarifai CLI provides several commands for local model development:

#### 1. Install the Clarifai Python SDK

Install the latest version of the Clarifai Python SDK:

```bash
pip install -U clarifai
```

#### 2. Login to Clarifai CLI

```bash
clarifai login
```

Fill the details requested during login


#### 3. Setup the Model Codebase

To setup a new codebase:

```bash
clarifai model init
```

Else if you have a codebase on git already:

```bash
clarifai model init <codebase-path> --github-repo <user>/<repo> --github-pat <github-token>
```

This sets up your codebase to develop.


#### 4. Start your Local Runner

Now, you can start your local runner:

```bash
clarifai model local-runner
```

This connects your local model to the Clarifai platform for testing while keeping the model running locally.



### Test your Local Runner

#### Set Clarifai PAT
Refer this [guide](https://docs.clarifai.com/control/authentication/pat/#how-to-create-a-pat-on-the-platform) on how to obtain one from the platform.

#### Inference using SDK
Once your runner is started, you can test it with a simple prediction:

```bash
import os
from clarifai.client import Model
from clarifai.runners.utils import data_types

os.environ["CLARIFAI_PAT"] = "YOUR_PAT_HERE"

model = Model("https://clarifai.com/<user-id>/local-dev-runner-app/models/local-dev-model",
    deployment_id = 'local-dev-deployment', # Only needed for dedicated deployed models
    base_url='https://api.clarifai.com',
 )

    
# Example model prediction from different model methods: 

response = model.predict(prompt="What is the future of AI?", number_of_letters=3)
print(response)
```



## Advanced Usage

### Custom Model Development
To create your own model:
1. Modify the `MyModel` class in `1/model.py`
2. Implement your custom logic in the model methods
3. Update the `config.yaml` with your model details
4. Test locally using the provided commands
5. Upload to Clarifai platform

### Adding Dependencies
To add additional Python packages:
1. Add them to `requirements.txt`
2. Run `pip install -r requirements.txt`
3. Import and use them in your model code

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Not Loading**: Check that your model class inherits from `ModelClass` and implements `load_model()`
3. **Configuration Errors**: Verify your `config.yaml` has valid user_id, app_id, and model_id values
4. **"inference_compute_info not found" Error**: Ensure your `config.yaml` includes the complete `inference_compute_info` section as shown in the configuration example

### Getting Help
- Check the [Clarifai Documentation](https://docs.clarifai.com/)
- Use `clarifai --help` for CLI command help
- Use `clarifai model --help` for model-specific commands
