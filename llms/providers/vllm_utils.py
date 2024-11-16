"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""


import random
import time
from typing import Any

import openai
from openai import OpenAI

vllm_api_key = "EMPTY"
vllm_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=vllm_api_key,
    base_url=vllm_api_base,
)

from tqdm.asyncio import tqdm_asyncio


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
def generate_from_vllm_openai_completion():
    raise NotImplementedError


@retry_with_exponential_backoff
def generate_from_vllm_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    answer: str = response.choices[0].message.content
    return answer


# @retry_with_exponential_backoff
# # debug only
# def fake_generate_from_openai_chat_completion(
#     messages: list[dict[str, str]],
#     model: str,
#     temperature: float,
#     max_tokens: int,
#     top_p: float,
#     context_length: int,
#     stop_token: str | None = None,
# ) -> str:
#     if "OPENAI_API_KEY" not in os.environ:
#         raise ValueError(
#             "OPENAI_API_KEY environment variable must be set when using OpenAI API."
#         )

#     answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
#     return answer

