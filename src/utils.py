import json
import os
from retry import retry
from openai import OpenAI
import re

# Gemini support
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@retry()
def generate(data):
    """
    Generate completion using specified model.
    Supports OpenAI, Claude, Llama, and Gemini models.
    """
    if os.path.isfile(data["output_file"]):
        with open(data["output_file"]) as f:
            for i in f.readlines():
                i = json.loads(i)
                if i['id'] == data["id"]:
                    return i["output"]
    
    # Handle Gemini models separately
    if 'gemini' in data['model'].lower():
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "langchain_google_genai not installed. Install with: pip install langchain-google-genai"
            )
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model=data["model"],
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=8192
        )
        
        # Format messages for Gemini
        formatted_messages = []
        for msg in data["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_messages.append(("system", content))
            elif role == "assistant":
                formatted_messages.append(("assistant", content))
            else:
                formatted_messages.append(("human", content))
        
        # Generate response
        response = llm.invoke(formatted_messages)
        result = response.content.strip()
        
        # Format output similar to OpenAI format
        stream = {
            "choices": [{
                "message": {
                    "content": result,
                    "role": "assistant"
                }
            }],
            "model": data["model"]
        }
    else:
        # Handle OpenAI-compatible models
        if 'llama' in data['model']:
            client = OpenAI(api_key=os.getenv('LLAMA_API_KEY'), base_url=os.getenv('LLAMA_BASE_URL'), timeout=60)
        elif 'claude' in data['model']:
            client = OpenAI(api_key=os.getenv('CLAUDE_API_KEY'), base_url=os.getenv('CLAUDE_BASE_URL'), timeout=60)
        else:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=60)

        if "generation_config" in data:
            if 'claude' in data['model']:
                n = data['generation_config']["n"]
                stream = client.chat.completions.create(
                    model=data["model"],
                    messages=data["messages"],
                    **data["generation_config"]
                ).to_dict()
                for _ in range(n-1):
                    this_stream = client.chat.completions.create(
                        model=data["model"],
                        messages=data["messages"],
                        **data["generation_config"]
                    ).to_dict()
                    stream["choices"].append(this_stream["choices"][0])
            else:
                stream = client.chat.completions.create(
                    model=data["model"],
                    messages=data["messages"],
                    **data["generation_config"]
                ).to_dict()
        else:
            stream = client.chat.completions.create(
                model=data["model"],
                messages=data["messages"],
            ).to_dict()
    
    output_data = {
        "id": data["id"],
        "output": stream,
        "input": data
    }
    write_jsonl(file_path=data["output_file"], single_data=output_data)
    
    return output_data["output"]

def read_json(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path) as f:
        return json.load(f)

def read_jsonl(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path) as f:
        result = [json.loads(line) for line in f.readlines()]
    result = sorted(result, key=lambda x: x['id'])
    return result

def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_jsonl(file_path, single_data):
    with open(file_path, 'a') as f:
        f.write(json.dumps(single_data, ensure_ascii=False)+'\n')


def find_value(input_string):
    pattern = r'\(\s*(0(\.\d+)?|1(\.0+)?)\s*\|\s*(.+?)\s*\)'
    matches = re.findall(pattern, input_string)

    values, evals = [], []
    for m in matches:
        pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
        tmp_score = pattern.findall(m[0])
        value = float(tmp_score[0])
        values.append(value)
        
        text = '|'.join(input_string.split('|')[1:])
        text = ')'.join(text.split(')')[:-1]).strip()

        evals.append(text)
    return values, evals