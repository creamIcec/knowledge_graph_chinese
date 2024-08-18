import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client


def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    );
    SYS_PROMPT_CHINESE = (
        "你的任务是提取在给定上下文中提到的关键概念（和非个人实体）。 "
        "只提取最重要的原子概念，如果需要，将概念分解为更简单的概念。" 
        "将概念分类为以下类别之一：" 
        "[事件、概念、地点、对象、文件、组织、条件、杂项]\n" 
        "使用以下格式将输出格式化为 json 列表：\n"
         "[\n"
         "  {\n"
         '      "entity": 概念，\n'
         '      "importance"：概念的上下文重要性，等级为 1 到 5(5 为最高),\n'
         '      "category"：概念的类型，\n'
         "  }, \n"
         "{ }, \n" 
         "] \n"
    );
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    );
    SYS_PROMPT_CHINESE = (
        "您是一位网络图制作者，可以从给定的上下文中提取术语及其关系。"
        "您将获得一个上下文块（以 ``` 分隔），您的任务是提取给定上下文中提到的术语的本体。这些术语应根据上下文代表关键概念。\n"
        "想法 1:在遍历每个句子时，思考其中提到的关键术语。\n"
            "\t术语可能包括对象、实体、位置、组织、人员、\n"
            "\t条件、首字母缩略词、文档、服务、概念等。\n"
            "\t术语应尽可能原子化\n\n"
        "想法 2:思考这些术语如何与其他术语具有一对一的关系。\n"
            "\t在同一个句子或同一个段落中提到的术语通常彼此相关。\n"
            "\t术语可以与许多其他术语相关\n\n"
        "想法 3:找出每个相关对之间的关​​系术语。\n\n"
        "将输出格式化为 JSON 列表。列表中的每个元素包含一对术语"
        "以及它们之间的关系，如下所示：\n"
        "[\n"
        "   {\n"
        '       "node_1": "从提取的本体中的概念",\n'
        '       "node_2": "从提取的本体中相关的概念",\n'
        '       "edge": "一两句话中两个概念 node_1 和 node_2 之间的关系"\n'
        "   }\n"
        "]"
    );

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
