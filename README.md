# VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks

<!-- <p align="center">
<a href="https://www.python.org/downloads/release/python-3109/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://mypy-lang.org/"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy"></a>
<a href="https://beartype.readthedocs.io"><img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" alt="bear-ified"></a>
</p> -->

[<a href="https://videowebarena.github.io">Website</a>] 
[<a href="https://arxiv.org/abs/2410.19100">Paper</a>]


VideoWebArena is an agent benchmark dedicated towards evaluating long context multimodal agents with video-based tasks. VideoWebArena tests agents' ability to take videos in-context and utilize them to complete realistic web tasks. VideoWebArena consists of 2,021 web agent tasks based on manually crafted video tutorials, which total almost four hours of content. 
For our benchmark, we define a taxonomy of long-context video-based agent tasks with two main areas of focus: skill retention and factual retention. It builds off the reproducible, execution based evaluation introduced in 
<a href="https://jykoh.com/vwa"> VisualWebArena</a> and <a href="https://webarena.dev" target="_blank">WebArena</a>. 

![Overview](media/overview.png)

## TODOs
- [x] Add website and arxiv links, update arxiv citation

## News
- [10/14/2024]: release of the VideoWebArena benchmark and codebase.

## Install
```bash
# Python 3.10 (or 3.11, but not 3.12 cause 3.12 deprecated distutils needed here)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```
You will need OpenAI Whisper to process the videos running the baseline agents. You download from the [their source repository](https://github.com/openai/whisper) .

Open a python shell and run the following commands to download the necessary models:
```python
import nltk
nltk.download('punkt')
```
You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```
pytest -x
```


## End-to-end Evaluation
1. Setup the standalone environments.
Please check out [this page](environment_docker/README.md) for details.

2. Configurate the urls for each website.
First, export the `DATASET` to be `videowebarena`:
```bash
export DATASET=videowebarena
```
Then, set the URL for the websites

```bash
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
```

3. Generate config files for each test example:
```bash
python scripts/generate_test_data.py
```
You will see `*.json` files generated in the [config_files](./config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:
```
bash prepare.sh
```

5. Set up API keys.

If using OpenAI models, set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
```
If using azure openai models, set a valid Azure API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
```
You will also need to pip install azure-identify to use azure openai models



If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install). Configure the API key by authenticating with Google Cloud:
```
gcloud auth login
gcloud config set project <your_project_name>
```
Then, set the following environment variables:
```
export VERTEXAI_LOCATION=<your_location>
export VERTEXAI_PROJECT=<your_project_name>
```

If using vllm, use the follow code to launch vllm server in a seperate terminal session
```
conda create -n vllm python=3.10 
conda activate vllm
pip install vllm
vllm serve <model name> --trust-remote-code --tensor-parallel-size <number of GPU> --limit-mm-per-prompt "image=100" --max-model-len <max token number supported by GPU memory>  --dtype <half if using old cuda verion>
# when error message says use float16, do --dtype half
# when error message says kv cache memory exceed memory, use --max-model-len to adjust length
vllm serve "microsoft/Phi-3.5-vision-instruct" --trust-remote-code  --tensor-parallel-size 4  --limit-mm-per-prompt "image=100" --max-model-len 32760 --dtype half # an example for 4*V100 GPU
```
6. Download the videos from this link: [videos](https://drive.google.com/file/d/17DwmsM7KzBWyz1BN1aq7NHDvgcTIrCgx/view?usp=sharing). You can use gdown to download the videos:
```bash
pip install gdown
gdown --id 17DwmsM7KzBWyz1BN1aq7NHDvgcTIrCgx
```
then move all videos to the media folder.



7. Launch the evaluation:

There are two types of agent in general for videowebarena evaluation:
- video_summary_agent (agent type name: video_summary_prompt): this type of agent first summarize the video into the text and then use the text as part of context during each step when agent takes action.
- video_agent (agent type name: video_prompt): this type of agent uses the entire video (or video frames) as part of context during each step when agent takes action.


There are two ways an LLM agent can process videos in-context:
- use video frames sampled from the video (e.g. gpt-4o)
- use entire video as input (e.g. gemini-pro)

More information can be found in our paper.

Video Frame Agent Evaluation Example: 

The video frame agent uses frames from the provided tutorial videos as input. In our paper, we use gpt-4o as the video frame agent. You can use the following command to run this evaluation:
```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_frame.json \ # this is the prompt file for video frame agent
  --test_start_idx <start_idx> \ # by default 0
  --test_end_idx <end_idx> \ #  by default 999
  --test_idx_ls 56 63 67\  # if you only want to test a list of tasks, provide id in this way and test start idx and test end idx will be ignored
  --test_config_base_dir config_files/videowa/test_classifieds \ # the config dir for which taskset to evaluate on
  --provider openai  \ # model_provider_name like openai, azopenai, google
  --model gpt-4o\ # model_name like gpt-4o, gemini-1.5-pro-001
  --action_set_tag som \ # no need to change
  --observation_type image_som\  # no need to change
  --result_dir "results"\ # directory to save the results
  --agent_type video_prompt\ # corresponds to which type of agent you are using
  --video_dir media\ # put downloaded videos here
  --max_frame_num 60\ # max number of frames sampled from the video
  --max_tokens 4096\ 
  --intermediate_intent_instruction_path agent/prompts/jsons/video_frame_intent_understanding.json # if this is present, the agent will also evaluate on intermediate intent understanding
```

This script will run the first Classifieds example with the GPT-4o video frame agent. The trajectory will be saved in <your_result_dir>/0.html. Note that the baselines that include a captioning model run on GPU by default (e.g., BLIP-2-T5XL as the captioning model will take up approximately 12GB of GPU VRAM). For parallel evaluation in one machine, it might be easier to load the model on CPU with the default setting `--eval_captioning_model_device cpu`.

If you'd like to reproduce the results from our paper, we have also provided scripts in `scripts/run_videowa.sh` to run the full evaluation pipeline on each of the VideoWebArena environments. For example, to reproduce the results from the Classifieds environment, you can run:

```bash
bash scripts/run_videowa.sh "test_classifieds" 0 100
```

We also support running the evaluation in parallel using the screen command!

```bash
bash scripts/parallel_run_start_end.sh
```




## Citation
If you find our environment or our models useful, please consider citing  <a href="videowebarena.github.io" target="_blank">VideoWebArena</a>, <a href="https://jykoh.com/vwa" target="_blank">VisualWebArena</a> as well as <a href="https://webarena.dev/" target="_blank">WebArena</a>:
```
@article{jang2024videowebarenaevaluatinglongcontext,
      title={VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks}, 
      author={Lawrence Jang and Yinheng Li and Charles Ding and Justin Lin and Paul Pu Liang and Dan Zhao and Rogerio Bonatti and Kazuhito Koishida},
      year={2024},
      eprint={2410.19100},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.19100}, 
}

@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}

@article{zhou2024webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={ICLR},
  year={2024}
}
```

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/web-arena-x/webarena">WebArena codebase and  the <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena codebase </a>.
