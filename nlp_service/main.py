from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
import re
import torch
#只能处理get请求
#python main.py
#可以在cmd 用命令 curl -G "http://localhost:8000/generate" --data-urlencode "question=如何打开客厅的灯？"测试
app = FastAPI()

# 初始化 DeepSeek 模型（生成初始回答）
llm = ChatOpenAI(
    temperature=0.7,
    model='deepseek-chat',
    api_key="sk-070151a6fcd14bed867ac165a2fce23a",  # 替换为你的实际API密钥
    base_url="https://api.deepseek.com"
)

# 加载本地微调的 BERT-BIO 分词模型
model_path = "./fine-tuned-home-bert"  # 假设模型已下载到本地
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 定义提示模板
prompt_template = """
你是一个家居智能助手,对于输入的问题
{question}
你应该输出JSON格式，元素要求如下：
{{
"output":"对于这个问题的智能回答",
"operation":"具体做了什么操作对什么家具(中文)"
}}
"""
prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])

def clean_json_response(response_str):
    """清理模型返回的JSON字符串"""
    response_str = re.sub(r'```json|```', '', response_str).strip()
    return response_str

def bio_tag_operation(operation_text: str):
    """使用微调模型对 operation 进行 BIO 分词"""
    # 分词和模型预测
    inputs = tokenizer(operation_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    
    # 获取 token 和对应的标签
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [model.config.id2label[pred] for pred in predictions]
    
    # 合并子词并提取实体
    entities = []
    current_entity = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": token.replace("##", ""), "type": label[2:]}
        elif label.startswith("I-"):
            if current_entity and current_entity["type"] == label[2:]:
                current_entity["text"] += token.replace("##", "")
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

@app.get("/generate")
async def generate_text(question: str):
    try:
        # 1. 调用 DeepSeek 生成初始回答
        formatted_prompt = prompt.format_messages(question=question)
        response = llm.invoke(formatted_prompt)
        cleaned_response = clean_json_response(response.content)
        result = json.loads(cleaned_response)

        # 2. 使用微调模型对 operation 进行 BIO 分词
        operation_text = result.get("operation", "")
        if operation_text:
            entities = bio_tag_operation(operation_text)
            result["operation_entities"] = entities  # 添加实体识别结果

        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
