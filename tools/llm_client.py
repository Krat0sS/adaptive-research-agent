"""LLM API 调用封装"""
import json
from openai import OpenAI
import config


def get_client():
    return OpenAI(api_key=config.OPENAI_API_KEY)


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """调用 LLM，返回文本响应"""
    client = get_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=config.OPENAI_MAX_TOKENS
    )
    return response.choices[0].message.content


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
    """调用 LLM，要求返回 JSON"""
    client = get_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt + "\n\n你必须返回合法的 JSON，不要包含任何其他文本。"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=config.OPENAI_MAX_TOKENS,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
