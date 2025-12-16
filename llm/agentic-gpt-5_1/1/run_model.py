# from clarifai.client.model import Model
# import numpy as np

# prompt = "scrape the clarifai website and summarize the content in 500 words using the tools provided"

# mcp_servers = [
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/slack-mcp-server/versions/cc33d8d8cb6845dab70fc1b57c9fe7ec",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/firecrawl-browser-tools-mcp-server/versions/81f4deef3d5f4c2187e44a8c98a9de2d",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/browser-tools-mcp-server/versions/8bca416b59964e73a2811dd1e852914d",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/browser-search-mcp/versions/098178b626e84105b5877c9758a00281",
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/browser-search-mcp-server/versions/7d42e2f792fc4bbfaf4e71b89aade071"
# ]

# model_url = "https://clarifai.com/luv_2261/local-runner-app/models/local-runner-model"
# model_prediction =  Model(url=model_url, pat="87c77b7b38f04901bbaf9ad9274a69cb").predict(prompt=prompt, 
#                                                                                           mcp_servers=mcp_servers)
# print(model_prediction)
# for res in model_prediction:
#     print(res, end="", flush=True)


# import asyncio
# import os
# from fastmcp import Client
# from fastmcp.client.transports import StreamableHttpTransport

# transport = StreamableHttpTransport(url="https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/browser-search-mcp/versions/098178b626e84105b5877c9758a00281",
#                                     headers={"Authorization": "Bearer " + "834cf345f0b248cb99cbb57fb6ea85ce"})

# async def main():
#   async with Client(transport) as client:
#     tools = await client.list_tools()
#     print(f"Available tools:\n\n {tools}")
#     # TODO: update the dictionary of arguments passed to call_tool to make sense for your MCP.
#     result = await client.call_tool("search", {'query': 'Clarifai latest news AI platform 2024', 'topn': 5, 'region': 'wt-wt'})
#     print(f"Result: {result[0].text}")

# if __name__ == "__main__":
#   asyncio.run(main())



# import os
# from clarifai.client import Model

# mcp_servers = [
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/calculator-mcp-server",
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/time-mcp-server",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/web-search-mcp-server",
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/mcp-server-weather"
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/github-mcp-server",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/calendar-mcp-server"
    
# ]
# model = Model(
#   url = "https://clarifai.com/luv_2261/local-runner-app/models/local-runner-model",
#   deployment_id = "local-runner-deployment", # Only needed for dedicated deployed models
#   deployment_user_id = "luv_2261"
# )

# Example model prediction from different model methods: 

# response = model.predict(
#     prompt = "what's the current time in NY?", 
#     mcp_servers = mcp_servers
# )
# print(response)

# response = model.generate(
#     prompt = "what's the current time in NY?", 
#     mcp_servers = mcp_servers
# )
# for res in response:
#     print(res, end="", flush=True)

# from openai import OpenAI

# import os
# client = OpenAI(
#     base_url="https://api.clarifai.com/v2/ext/openai/v1",
#     api_key= os.environ['CLARIFAI_PAT']
# )

# mcp_servers = [
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/calculator-mcp-server",
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/time-mcp-server",
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/browser-mcp-server",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/mcp-server-weather"
#     "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/github-mcp-server",
#     # "https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai/apps/mcp/models/calendar-mcp-server"
# ]

# messages = [
#     {"role": "user", "content": "Find my last 2 open PRs in clarifai/clarifai-python repository on github"}
#     ]
# completion = client.chat.completions.create(
#     model="https://clarifai.com/clarifai/agentic-model/models/gpt-5_1",
#     messages=messages,
#     extra_body={"mcp_servers": mcp_servers},
#     max_completion_tokens=10000,
#     stream=True
# )

# # print(completion)

# for chunk in completion:
#     if chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     elif chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # else:
    #     print(chunk, flush=True)
    
# print("Response API--------------------------------")
# resp = client.responses.create(
#     model="https://clarifai.com/luv_2261/local-runner-app/models/local-runner-model",
#     input="what is clarifai latest news?",
#     # tools=tools,
#     # tool_choice="auto"
#     extra_body={"mcp_servers": mcp_servers},
# )
# print(resp.output)

# Streaming Response API


# with client.responses.stream(
#     model="https://clarifai.com/luv_2261/local-runner-app/models/local-runner-model",
#     input="what is clarifai latest news?",
#     extra_body={"mcp_servers": mcp_servers},
# ) as stream:
#     print("Assistant:", end=" ", flush=True)

#     for event in stream:
#         # text deltas appear as this event type:
#         if event.type == "response.output_text.delta":
#             print(event.delta, end="", flush=True)

#     # after stream ends, you can get the final structured response:
#     final_response = stream.get_final_response()

# print("\n\nFinal Response Object:")
# print(final_response)
    