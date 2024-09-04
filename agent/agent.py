import argparse
import json
from typing import Any, Optional, Union
import os

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn=None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if (
            "gemini" in lm_config.model
            or "gpt-4" in lm_config.model
            and "vision" in lm_config.model
        ) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(trajectory, intent, meta_data)
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            if output_response:
                print(f"Agent: {response}", flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


class VideoUnderstanding:
    def __init__(
        self,
        lm_config: lm_config.LMConfig,
        prompt_constructor: Union[VideoFrameUnderstandingPromptConstructor, VideoFramePromptConstructor]
    ):
        self.prompt_constructor = prompt_constructor
        self.lm_config = lm_config

    def get_video_summary(self, video_path: str, intent: str):
        prompt = self.prompt_constructor.construct(video_path, intent)
        response = call_llm(self.lm_config, prompt)
        print(f"Video Path: {video_path}\nVideo Summary: {response}", flush=True)
        return response
    
    def get_video_path(self, args: argparse.Namespace, config_file: str):
        config = json.load(open(config_file))
        video_name = config["video"]
        video_path = os.path.join(args.video_dir, video_name + ".mov")
        return video_path

    def get_intermidiate_intent(self, args: argparse.Namespace, config_file: str):
        config = json.load(open(config_file))
        intent = config["intermediate_intent"]
        video_path = self.get_video_path(args, config_file)
        response = self.get_video_summary(video_path, intent)
        return response


class VideoPromptAgent(PromptAgent):
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: Union[VideoFramePromptConstructor, VideoPromptConstructor],
        video_dir: str,
    ):
        super().__init__(action_set_tag, lm_config, prompt_constructor)
        self.video_dir = video_dir
        assert (
            "gpt-4o" in lm_config.model or "gemini" in lm_config.model
        ), "VideoPromptAgent only supports gpt-4o model and gemini model"

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
    ) -> Action:
        # Create page screenshot image for multimodal models.
        page_screenshot_arr = trajectory[-1]["observation"]["image"]
        page_screenshot_img = Image.fromarray(
            page_screenshot_arr
        )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")

        prompt = self.prompt_constructor.construct(
            trajectory, intent, page_screenshot_img, images, self.video_path, meta_data
        )
        # prompt = self.prompt_constructor.construct(
        #     trajectory, intent, meta_data
        # )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            if output_response:
                print(f"Agent: {response}", flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:  # add video_path
        config = json.load(open(test_config_file))
        self.video_name = config["video"]
        self.video_path = os.path.join(self.video_dir, self.video_name + ".mov")


class VideoSummaryPromptAgent(PromptAgent):
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: VideoSummaryPromptConstructor,
        video_dir: str,
        video_prompt_constructor: Union[VideoFrameUnderstandingPromptConstructor, VideoUnderstandingPromptConstructor],
    ):
        super().__init__(action_set_tag, lm_config, prompt_constructor)
        self.video_dir = video_dir
        self.video_understanding = VideoUnderstanding(
            lm_config, video_prompt_constructor
        )
        assert (
            ("gpt-4o" in lm_config.model or "gemini" in lm_config.model)
        ), "VideoSummaryPromptAgent only supports gpt-4o model and gemini model"

    def reset(self, test_config_file: str) -> None:
        config = json.load(open(test_config_file))
        self.video_name = config["video"]
        self.video_path = os.path.join(self.video_dir, self.video_name + ".mov")
        self.video_summary = self.video_understanding.get_video_summary(
            self.video_path, config["intent"]
        )

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
    ) -> Action:
        # Create page screenshot image for multimodal models.
        page_screenshot_arr = trajectory[-1]["observation"]["image"]
        page_screenshot_img = Image.fromarray(
            page_screenshot_arr
        )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"

        prompt = self.prompt_constructor.construct(
            trajectory,
            intent,
            self.video_summary,
            page_screenshot_img,
            images,
            meta_data,
        )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            # raise ValueError(response)
            if output_response:
                print(f"Agent: {response}", flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
        )
    elif args.agent_type == "video_prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        if constructor_type == "VideoFramePromptConstructor":
            prompt_constructor = eval(constructor_type)(
                args.instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
                max_frame_num=args.max_frame_num,
            )
        elif constructor_type == "VideoPromptConstructor":
            prompt_constructor = eval(constructor_type)(
                args.instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
        )
        else:
            raise ValueError(f"Unsupported video prompt constructor type {constructor_type}")
        agent = VideoPromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            video_dir=args.video_dir,
        )
    elif args.agent_type == "video_summary_prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        with open(args.video_summary_instruction_path) as f:
            video_constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            instruction_path=args.instruction_path,
            lm_config=llm_config,
            tokenizer=tokenizer,
        )
        
        if video_constructor_type == "VideoFrameUnderstandingPromptConstructor":
            video_prompt_constructor = VideoFrameUnderstandingPromptConstructor(
                instruction_path=args.video_summary_instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
                max_frame_num=args.max_frame_num,
            )
        elif video_constructor_type == "VideoUnderstandingPromptConstructor":
            video_prompt_constructor = VideoUnderstandingPromptConstructor(
                instruction_path=args.video_summary_instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported video summary prompt constructor type {video_constructor_type}")
        agent = VideoSummaryPromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            video_dir=args.video_dir,
            video_prompt_constructor=video_prompt_constructor,
        )
    else:
        raise NotImplementedError(f"agent type {args.agent_type} not implemented")
    return agent


def construct_intermediate_intent_agent(args: argparse.Namespace) -> Agent:
    llm_config = lm_config.construct_llm_config(args)
    agent: Agent
    if args.intermediate_intent_instruction_path:
        with open(args.intermediate_intent_instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        if constructor_type == "VideoFrameUnderstandingPromptConstructor":
            prompt_constructor = eval(constructor_type)(
                instruction_path=args.intermediate_intent_instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
                max_frame_num=args.max_frame_num,
            )
        elif constructor_type == "VideoUnderstandingPromptConstructor":
            prompt_constructor = eval(constructor_type)(
                instruction_path=args.intermediate_intent_instruction_path,
                lm_config=llm_config,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported video prompt constructor type {constructor_type}")
        agent = VideoUnderstanding(
                lm_config=llm_config,
                prompt_constructor=prompt_constructor,
            )

    else:
        raise ValueError("Intermediate intent instruction path is not provided")
    return agent