import time
import os
from typing import Dict, Union

import anthropic
import openai
import tiktoken
from whale.util import Timeout
from whale import TextGeneration

def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret


def create_whale_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "whale-2.0",
) -> Dict:
    DEFAULT_CONNECT_TIMEOUT = 120  # 连接超时
    DEFAULT_READ_TIMEOUT = 120     # 读取超时
    if isinstance(message, list):
        config = {
            "model": model,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "n": batch_size,
            "timeout": Timeout(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "n": batch_size,
            "timeout": Timeout(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
            "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": message}],
        }
    return config


# def request_whale_engine(config, logger, base_url="https://whale-wave.alibaba-inc.com", max_retries=4, timeout=100):
#     ret = None
#     retries = 0

#     # 设置API key
#     TextGeneration.set_api_key("JUZJ8885DG", base_url=base_url)
#     # 直接使用TextGeneration类，不需要创建实例
#     client = TextGeneration

#     while ret is None and retries < max_retries:
#         try:
#             # Attempt to get the completion
#             logger.info("Creating API request")
            
#             # 请求模型
#             response = client.chat(
#                 model=config["model"],
#                 messages=config["messages"],
#                 stream=config["stream"],
#                 temperature=config["temperature"],
#                 max_tokens=config["max_tokens"],
#                 timeout=config["timeout"],
#                 top_p=config["top_p"],
#                 n=config["n"]  # 添加采样数量参数
#             )

#             # 处理流式结果
#             results = ["" for _ in range(config["n"])]  # 初始化多个结果列表
#             for event in response:
#                 if event.error_code is not None:
#                     logger.error(f'Error: {event.error_code, event.message}')
#                     raise Exception(f"API Error: {event.error_code} - {event.message}")
#                 else:
#                     if event.choices[0].finish_reason is not None and event.choices[0].finish_reason != '':
#                         logger.info(f'Finished Reason: {event.choices[0].finish_reason}')
#                         break
                    
#                     # 处理多个选择
#                     for i, choice in enumerate(event.choices):
#                         if i < len(results):  # 确保不超出结果列表范围
#                             content = choice.delta.content
#                             results[i] += content

#             ret = results
#             logger.info("Successfully got response from API")

#         except Exception as e:
#             logger.error("Unknown error. Waiting...", exc_info=True)
#             time.sleep(10 * retries)
#         retries += 1

#     logger.info(f"API response {ret}")
#     return ret


def request_whale_engine(config, logger, base_url="https://whale-wave.alibaba-inc.com", max_retries=4, timeout=100):
    # 设置API key
    TextGeneration.set_api_key("JUZJ8885DG", base_url=base_url)
    # 直接使用TextGeneration类，不需要创建实例
    client = TextGeneration
    
    results = []
    num_samples = config["n"]
    
    # 对每个样本单独进行请求
    for sample_idx in range(num_samples):
        ret = None
        retries = 0
        
        while ret is None and retries < max_retries:
            try:
                # Attempt to get the completion
                logger.info(f"Creating API request for sample {sample_idx + 1}/{num_samples}")
                
                # 请求模型 - 每次只请求一个样本
                response = client.chat(
                    model=config["model"],
                    messages=config["messages"],
                    stream=config["stream"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"],
                    top_p=config["top_p"]
                )

                # 处理流式结果
                current_result = ""
                for event in response:
                    if event.error_code is not None:
                        logger.error(f'Error: {event.error_code, event.message}')
                        raise Exception(f"API Error: {event.error_code} - {event.message}")
                    else:
                        if event.choices[0].finish_reason is not None and event.choices[0].finish_reason != '':
                            logger.info(f'Finished Reason: {event.choices[0].finish_reason}')
                            break
                        
                        content = event.choices[0].delta.content
                        current_result += content

                ret = current_result
                logger.info(f"Successfully got response for sample {sample_idx + 1}")

            except Exception as e:
                logger.error(f"Error for sample {sample_idx + 1}. Waiting...", exc_info=True)
                time.sleep(10 * retries)
            retries += 1
            
        if ret is not None:
            results.append(ret)
        else:
            logger.error(f"Failed to get response for sample {sample_idx + 1} after {max_retries} retries")
            results.append("")  # 添加空字符串作为失败的结果

    logger.info(f"API responses: {results}")
    return results