"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import json
import urllib
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
az_endpoint = os.environ["AZURE_ENDPOINT"]
az_api_key = os.environ["AZURE_API_KEY"]
azclient = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=az_endpoint,
    azure_ad_token_provider=get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
)
az_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {az_api_key}",
        }


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper



@retry_with_exponential_backoff
def generate_from_azure_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "phi3v" in model:
        return get_chat_completion_phi(messages, temperature, max_tokens, top_p, context_length, stop_token)
    elif "gpt" in model:
        return get_chat_completion_gpt(messages, model, temperature, max_tokens, top_p, context_length, stop_token)
    else:
        raise NotImplementedError

def get_chat_completion_gpt(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
) -> str:
    response = azclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    answer: str = response.choices[0].message.content
    return answer

def get_chat_completion_phi(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
    ) -> str:
    data = {
        "input_data": {
            "input_string": messages,
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}
        }
    }
    body = str.encode(json.dumps(data))
    req = urllib.request.Request(az_endpoint, body, az_headers)

    response = json.load(urllib.request.urlopen(req))
    return response["output"]