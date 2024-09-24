source scripts/set_configs.sh

export domain=$1
export test_config_base_dir="$test_config_dir/$domain"
export result_dir="results/$DATASET/$domain/video_prompt_frame" 
export video_domain=${domain:5}  





# video frame prompt with intermediate eval
rm -rf $result_dir
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_frame.json \
  --test_start_idx=$test_start_idx \
  --test_end_idx=$test_end_idx \
  --test_config_base_dir=$test_config_base_dir \
  --provider=$videoframe_model_provider \
  --model=$videoframe_model\
  --action_set_tag som \
  --observation_type image_som\
  --result_dir $result_dir\
  --agent_type video_prompt\
  --video_dir media\
  --max_frame_num=$max_frame_num\
  --max_tokens 4096\
  --intermediate_intent_instruction_path agent/prompts/jsons/video_frame_intent_understanding.json

## video frame summary
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_summary.json \
#   --video_summary_instruction_path agent/prompts/jsons/video_frame_understanding.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$videoframe_model_provider \
#   --model=$videoframe_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_summary_prompt\
#   --video_dir media\
#   --max_tokens 4096\
#   --max_frame_num $max_frame_num
  # --intermediate_intent_instruction_path agent/prompts/jsons/video_frame_intent_understanding.json


# video prompt with intermediate eval
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$video_model_provider \
#   --model=$video_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_prompt\
#   --video_dir media\
#   --mode completion\
#   --max_tokens 8000\
#   --intermediate_intent_instruction_path agent/prompts/jsons/video_intent_understanding.json


# video summary 
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_summary.json \
#   --video_summary_instruction_path agent/prompts/jsons/video_understanding.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$video_model_provider \
#   --model=$video_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_summary_prompt\
#   --video_dir media\
#   --mode completion\
#   --max_tokens 8000\
  # --intermediate_intent_instruction_path agent/prompts/jsons/video_intent_understanding.json






