# Import relevant functionality
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_tavily import tavily_search
from tavily import TavilyClient

import os
import ast
import re
from dotenv import load_dotenv  # 导入dotenv库
# 加载.env文件中的环境变量（需放在代码最顶部）
load_dotenv()  # 自动读取项目根目录的.env文件

# Create the agent
memory = MemorySaver()
model = ChatOllama(model="qwen2.5:7b")
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_agent(
    model,
    tools,
    checkpointer=memory,
    system_prompt="You are a helpful assistant. You MUST use the search tool to find current information. Never make up search results or fake data."
)
# client = TavilyClient("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")
# response = client.search(query="weather in San Francisco")
# print(response)
# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
