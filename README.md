# SGLang Model Serving Template

This is a template for serving models using SGLang.

## Attention Backend

The default attention backend is **`torch_native`**.

To use other backends (flashinfer, triton, etc.):
1. Modify `requirements.txt` with the required dependencies
2. Build your own Dockerfile with the necessary backend dependencies
3. Upload the model via CLI and specify your customized Dockerfile

Note: uploading the model via CLI will allow you to obtain the default generated Dockerfile. You can then customize the Dockerfile and re-upload again (CLI will provide you the option to use your custom Dockerfile)