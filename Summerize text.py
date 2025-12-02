import os
import ast
import re
from dotenv import load_dotenv  # 导入dotenv库
# 加载.env文件中的环境变量（需放在代码最顶部）
load_dotenv()  # 自动读取项目根目录的.env文件
import operator
import asyncio

from typing import Annotated, List, Literal, TypedDict,Iterable,Callable
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_classic.chains import LLMChain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic import hub
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

token_max = 1000
def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

def split_list_of_docs(docs: Iterable[Document], length_function: Callable, token_max: int) -> List[List[Document]]:
    docs = list(docs)
    doc_lists = []
    current_list = []
    current_length = 0
    for doc in docs:
        doc_length = length_function([doc])
        if current_length + doc_length <= token_max:
            current_list.append(doc)
            current_length += doc_length
        else:
            doc_lists.append(current_list)
            current_list = [doc]
            current_length = doc_length
    if current_list:
        doc_lists.append(current_list)
    return doc_lists

async def acollapse_docs(docs: Iterable[Document], reduce_func: Callable) -> Document:
    docs = list(docs)
    # 拼接所有文档内容作为 Prompt
    content = "\n\n".join(d.page_content for d in docs)
    # 调用 LLM 生成合并后的摘要
    summary = await reduce_func({"input": content})
    # 构造新的 Document 返回
    return Document(page_content=summary, metadata={"source": "collapsed"})

class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str

# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]
# collect summeries
def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

async def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content

# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}

# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    text = "\n\n".join(d.page_content for d in state["collapsed_summaries"])
    response = await _reduce({"input": text})   # 确保调用时为 dict
    return {"final_summary": response}

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)
map_prompt = hub.pull("rlm/map-prompt")

# Also available via the hub: `hub.pull("rlm/reduce-prompt")`
reduce_template = """
The following is a set of summaries:
{input}
Take these and distill it into a final, consolidated summary
of the main themes,and limit token is 50,can not surpass this token
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])



llm = ChatOllama(model="qwen2.5:7b")
loader = WebBaseLoader("https://blog.csdn.net/aifs2025/article/details/153332260?spm=1000.2115.3001.10525")
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)


# Construct the graph
# Nodes:
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

# Edges:
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()
# 正确调用图的示例（关键点：初始 state 必须和 OverallState 定义一致）
async def main():
    initial_state = {
        "contents": [d.page_content for d in split_docs],
        "summaries": [],
        "collapsed_summaries": [],
        "final_summary": "",
    }
# 流式调用（可观察中间更新）
 # 异步 invoke（等待整个图执行完）
    res = await app.ainvoke(initial_state)
    print("final:", res["final_summary"])

if __name__ == "__main__":
    asyncio.run(main())



# Instantiate chain
# chain = create_stuff_documents_chain(llm, prompt)
#
# result = chain.invoke({"context": docs})
#
# for token in chain.stream({"context": docs}):
#     print(token, end=" ")
# print(result)