from openai import OpenAI
import os
client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT'],
)
import time


# 119512 characters total
# 26054 tokens total
long_context = ""
with open("man-bash.txt", "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len 16384
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_context = long_context[:70000]

question = "Summarize the long context above in 1000 words."
prompt = f"{long_context}\n\n{question}"

t1 = time.time()
response = client.chat.completions.create(
    model="https://clarifai.com/luv_2261/test-upload/models/vllm-lmcache-kvcache-offloading-llama-3_2-1b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt,
        },
    ],
    temperature=0.7,
    stream=False, # stream=True also works, just iterator over the response
    max_tokens=512,
)
t2 = time.time()
print(f"Time taken: {t2 - t1} seconds")
print(response)