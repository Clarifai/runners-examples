# from clarifai.client import Model
# import time
# model = Model("https://web-dev.clarifai.com/luv_2261/test-upload/models/vllm_llama-3_2-1B-instruct-amd",
#                deployment_id = 'deploy-vllm_llama-3_2-1b-instruct-amd', # Only needed for dedicated deployed models
               
#  )  


# while True:
#     try:
#         # Example model prediction from different model methods: 
#         # response = model.predict(prompt='what is api?', max_tokens=5)
#         # print(response)

#         # Generate a response
#         res = model.predict(prompt='what is api?', max_tokens=5, temperature=0.7, top_p=0.8)
#         print(res)  # Print the response
#         break  # Exit the loop if successful
#     except Exception as e:
#         print(f"Error occurred: {e}. Retrying...")  # Handle the error and retry
#         time.sleep(30)  # wait for 30 seconds before retrying
        

from clarifai.client import Model
model = Model(
    "https://web-dev.clarifai.com/luv_2261/test-upload/models/vllm-gpt-oss-120b-amd",
    deployment_id = 'deploy-vllm-gpt-oss-120b-amd',
    pat="bf716abc8a8742818e9a1dc645fe8a47",
    base_url="https://api-dev.clarifai.com",
    deployment_user_id="luv_2261",
 )
res = model.predict(prompt='what is api?',)
print(res)
