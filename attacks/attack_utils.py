import json
import re

def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt

def fix_common_json_issues(json_str: str) -> str:
    # 去掉前后非 JSON 内容（如解释性语句）
    json_str = json_str.strip()

    # 删除 JSON 结尾可能出现的多余逗号（如最后一个数组元素后）
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # 确保 JSON 最外层闭合
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if close_braces < open_braces:
        json_str += '}' * (open_braces - close_braces)

    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if close_brackets < open_brackets:
        json_str += ']' * (open_brackets - close_brackets)

    return json_str

def parse_json(output):
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        fixed_output = fix_common_json_issues(output)
        try:
            data = json.loads(fixed_output)
            return data
        except json.JSONDecodeError as e:
            print("parse adjusted json", e)
        return None
    
def get_response(model, query):
    
    if isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    elif isinstance(query, list):
        messages = query
    else:
        raise ValueError("Query Format Error!") 

    for _ in range(3):
        try:
            resp = model.generate_response(messages)
            return resp
        except Exception as e: 
            print(f"Model Call Error: {e}")
            continue

    return ""

def get_response_append(model, dialog_hist, query):
    dialog_hist.append({"role": "user", "content": query})
    resp = get_response(model, dialog_hist)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist