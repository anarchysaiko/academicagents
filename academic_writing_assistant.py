import os
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


# 从环境变量或用户输入获取API密钥、基础URL和模型名称
def get_config():
    api_key = input("请输入API密钥 (直接回车使用环境变量): ").strip() or os.getenv(
        "API_KEY"
    )
    api_base = input(
        "请输入API基础URL (直接回车使用环境变量或默认值): "
    ).strip() or os.getenv("API_BASE")
    model = input(
        "请输入模型名称 (直接回车使用环境变量或默认值): "
    ).strip() or os.getenv("API_MODEL")

    if not api_key:
        raise ValueError("API密钥未提供。请通过输入或环境变量 API_KEY 设置API密钥。")

    return api_key, api_base, model


# 获取配置
api_key, api_base, model = get_config()

# 初始化OpenAI模型
llm = ChatOpenAI(
    temperature=0.7,
    api_key=api_key,
    base_url=api_base,
    model_name=model,
)


# 定义智能体及其提示模板
agents = [
    {
        "name": "研究领域专家",
        "prompt": "你是一位研究领域专家。根据用户的专业领域和学术兴趣，确定研究领域。请给出一个宽泛但明确的研究领域。用户输入: {user_input}",
    },
    {
        "name": "研究对象分析师",
        "prompt": "你是一位研究对象分析师。根据确定的研究领域'{previous_output}'和用户输入，确定具体的研究对象。研究对象应该比研究领域更加具体和聚焦。例如，对于社会保障专业，如果研究领域是'民营养老'，那么研究对象可能是'民营养老机构的定价机制'。请根据这个例子，为用户确定一个具体的研究对象。用户输入: {user_input}",
    },
    {
        "name": "问题洞察专家",
        "prompt": "你是一位问题洞察专家。根据确定的研究对象'{previous_output}'和用户输入，揭示本质问题。这个过程需要通过不断发现'现象问题'，寻找出现这种'现象问题'的原因，最终得到本质问题。例如，对于公共管理专业，'上海市农民工社会融入存在困境'是'现象问题'，而'超大城市原住民与外来人口的社会排斥'则是'本质问题'。请确保你揭示的本质问题表述简洁、明确，并使用名词性结构。用户输入: {user_input}",
    },
    {
        "name": "研究论题专家",
        "prompt": "你是一位研究论题专家。根据揭示的本质问题'{previous_output}'和用户输入，形成研究的论题。请遵循以下步骤：1) 提出基本判断；2) 构建理论框架，包括了解相关理论的核心概念、主要观点、假设前提以及发展历程；3) 分析理论的适应性；4) 根据理论框架，进一步明确研究问题和研究目标。最后，请提出一个清晰的研究论题。用户输入: {user_input}",
    },
    {
        "name": "论文题目专家",
        "prompt": "你是一位论文题目专家。根据形成的研究论题'{previous_output}'和用户输入，凝练论文题目。你的题目应该做到：1) 一眼就能看出研究对象和研究问题；2) 简洁明了；3) 吸引读者兴趣。请提供一个符合学术规范的论文题目。用户输入: {user_input}",
    },
]

chat_history = []


def agent_response(agent, user_input, previous_output):
    prompt = ChatPromptTemplate.from_template(agent["prompt"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(user_input=user_input, previous_output=previous_output)
    chat_history.append(f"{agent['name']}: {response}")
    return response


def academic_writing_assistant(user_input):
    responses = []
    previous_output = user_input

    for agent in agents:
        response = agent_response(agent, user_input, previous_output)
        formatted_response = f"【{agent['name']}】\n{response}\n"
        responses.append(formatted_response)
        previous_output = response

    chat_history.extend(responses)
    return "\n".join(responses)


# 创建Gradio界面
iface = gr.Interface(
    fn=academic_writing_assistant,
    inputs=gr.Textbox(label="用户输入", placeholder="请输入您的专业领域和研究兴趣..."),
    outputs=gr.Textbox(label="智能体回复", lines=20),
    title="学术论文写作助手",
    description="这是一个交互式学术论文写作助手。请输入您的专业领域和研究兴趣，智能体将按顺序帮助您确定研究领域、研究对象、本质问题、研究论题，并最终凝练论文题目。",
    allow_flagging="never",
)

# 启动Gradio应用
iface.launch()
