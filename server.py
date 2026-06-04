from flask import Flask, request, jsonify
import requests
import csv
import os
import random
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain_core.tools import tool

CURRENT_PREDICTION = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

CSV_FILE = "edge_ai_results.csv"

HISTORY_FILE = os.path.join(BASE_DIR, "diagnosis_history.csv")

@tool
def search_current_disease_knowledge() -> str:
    """查询当前识别出的植物病害类别对应的本地知识库资料。无需传入参数。"""

    print("\n===== Knowledge Tool Called =====")
    print("Current Prediction:", CURRENT_PREDICTION)
    print("=================================\n")

    if not CURRENT_PREDICTION:
        return "当前没有可用的识别结果。"

    return get_disease_knowledge(CURRENT_PREDICTION)


@tool
def get_weather_context() -> str:
    """获取当前天气信息。目前为预留接口或模拟天气。"""
    return get_weather_info()


@tool
def get_recent_history() -> str:
    """查询最近几条诊断历史记录，用于参考。"""
    print("\n===== History Tool Called =====\n")
    try:
        if not os.path.exists(HISTORY_FILE):
            return "暂无历史诊断记录。"

        with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        recent = lines[-5:]
        return "".join(recent)

    except Exception as e:
        return f"历史记录读取失败：{str(e)}"

LLM_MODEL = "qwen2.5:3b" 

agent_model = ChatOllama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
    temperature=0.3
)

plant_agent = create_agent(
    model=agent_model,
    tools=[
        search_current_disease_knowledge,
        get_weather_context,
        get_recent_history
    ],
    system_prompt="""
你是一名植物病害智能诊断 Agent。

你可以使用工具：
1. search_disease_knowledge：查询病害知识库
2. get_weather_context：获取天气或环境信息
3. get_recent_history：查询历史诊断记录

工作要求：
1. 必须优先查询病害知识库
2. 当需要查询病害知识库时，必须使用 search_current_disease_knowledge。该工具不需要参数。不要自行翻译、改写或猜测病害类别名。
3. 结合天气信息判断病害扩散风险也是重要的一部分
4. 回答之前，可以参考历史记录，也许可以减少耗时
5. 最终回答必须是中文
6. 回答适合手机APP展示
7. 不要输出工具调用过程

最终输出格式：

【风险等级】
低/中/高/极高

【天气影响判断】

【病害描述】

【当前处理建议】

【预防措施】
"""
)

def save_diagnosis_history(data, treatment):
    file_exists = os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time",
                "model",
                "prediction",
                "confidence",
                "inference_time_ms",
                "fps",
                "model_size_mb",
                "memory_mb",
                "battery",
                "estimated_energy_mj",
                "treatment"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("model", ""),
            data.get("prediction", ""),
            data.get("confidence", ""),
            data.get("inference_time_ms", ""),
            data.get("fps", ""),
            data.get("model_size_mb", ""),
            data.get("memory_mb", ""),
            data.get("battery", ""),
            data.get("estimated_energy_mj", ""),
            treatment
        ])

def get_disease_knowledge(prediction):
    file_path = os.path.join(KNOWLEDGE_DIR, prediction + ".txt")

    if not os.path.exists(file_path):
        return "暂无该类别的专门知识库资料，请根据通用植物病害防治原则回答。"

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_weather_info():
    weather_list = [
        """
    当前天气：晴
    温度：30℃
    湿度：55%
    未来24小时降雨概率：10%
    """,
        """
    当前天气：小雨
    温度：26℃
    湿度：88%
    未来24小时降雨概率：75%
    """,
        """
    当前天气：暴雨
    温度：24℃
    湿度：95%
    未来24小时降雨概率：95%
    """
    ]

    return random.choice(weather_list)

def generate_treatment(prediction,use_weather=False):
    global CURRENT_PREDICTION

    CURRENT_PREDICTION = prediction

    if use_weather:
        weather_instruction = "本次诊断启用天气因素，请调用天气工具并结合天气分析。"
    else:
        weather_instruction = "本次诊断不启用天气因素，不要调用天气工具，不要讨论外部天气影响。"

    try:
        result = plant_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
当前模型识别出的植物病害类别是：

{prediction}

{weather_instruction}

请调用工具查询当前病害知识库，生成诊断建议。

注意：
1. 不要翻译或修改类别名
2. 查询知识库时不需要自己输入病名
3. 必须调用 search_current_disease_knowledge 工具
4. 最终只输出诊断建议
"""
                }
            ]
        })

        messages = result["messages"]
        return messages[-1].content

    except Exception as e:
        return f"AI处理建议生成失败：{str(e)}"

@app.route("/upload", methods=["POST"])
def upload():

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "failed",
            "message": "no json"
        }), 400

    prediction = data.get("prediction", "unknown")

    use_weather = data.get("use_weather", False)

    treatment = generate_treatment(prediction, use_weather)

    save_diagnosis_history(data, treatment)

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=[
            "timestamp",
            "model",
            "prediction",
            "confidence",
            "inference_time_ms",
            "fps",
            "model_size_mb",
            "memory_mb",
            "battery",
            "estimated_energy_mj"
        ])

        if not file_exists:
            writer.writeheader()

        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow(data)

    print("\n========================")
    print("Prediction:", prediction)
    print("========================")
    print(treatment)
    print("========================\n")

    response = jsonify({
    "status": "success",
    "prediction": prediction,
    "treatment": treatment
    })
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)