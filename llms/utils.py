import argparse
from typing import Any

import os
import base64

try:
    from vertexai.generative_models import GenerativeModel, Part
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_from_azure_chat_completion,
    lm_config,
)

try:
    import anthropic
    from llms import generate_from_anthropic_chat_completion
except:
    print('Anthropic not set up, skipping import of providers.anthropic_utils.generate_from_anthropic_chat_completion')


APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "anthropic":
        assert isinstance(prompt, list)
        response = generate_from_anthropic_chat_completion(
            messages=prompt,
            model=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            context_length=lm_config.gen_config["context_length"],
            max_tokens=lm_config.gen_config["max_tokens"],
            stop_token=None,
        )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    elif lm_config.provider == "azure":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_azure_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        else:
            raise ValueError(
                f"Azure models do not support mode {lm_config.mode}"
            )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented, please choose from 'openai', 'huggingface', 'google', or 'azure'"
        )
    return response



def load_and_encode_video(video_path: str, provider: str):
    if provider == "google":
        with open(video_path, 'rb') as video_file:  
            video_data = video_file.read()  
            base64_video = base64.b64encode(video_data)  
        ext = os.path.splitext(video_path)[1][1:].lower()
        video_part = Part.from_data(
            mime_type=f"video/{ext}",
            data=base64.b64decode(base64_video))
        return video_part
    else:
        raise NotImplementedError(
            f"Provider {provider} not implemented"
        )