# SGLang Model Serving Template

This is a template for serving models using SGLang.

## Dockerfile

When uploading your model via CLI:
1) make sure you have clarifai and huggingface_hub installed.
2) choose the option to use your existing custom Dockerfile. This leverages the SGLang prepared base image that includes all necessary dependencies.