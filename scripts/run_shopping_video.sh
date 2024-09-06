
# old script
# python run.py \
#   --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
#   --test_start_idx 0 \
#   --test_end_idx 3 \
#   --test_config_base_dir=config_files/videowa/test_shopping \
#   --provider azopenai \
#   --model gpt-4o \
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir results


# python run.py \
#   --instruction_path agent/prompts/jsons/p_multimodal_cot_id_actree_3s_vid_summary.json \
#   --video_summary_instruction_path agent/prompts/jsons/p_cot_video_screenshots.json \
#   --test_start_idx 0 \
#   --test_end_idx 6 \
#   --test_config_base_dir=config_files/videowa/test_shopping \
#   --provider azopenai \
#   --model gpt-4o \
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir results/video_frame_summary\
#   --agent_type video_summary_prompt\
#   --video_dir media\
#   --max_frame_num 60

# python run.py \
#   --instruction_path agent/prompts/jsons/p_multimodal_cot_id_actree_3s_vid.json \
#   --test_start_idx 0 \
#   --test_end_idx 6 \
#   --test_config_base_dir=config_files/videowa/test_shopping \
#   --provider azopenai \
#   --model gpt-4o-mini \
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir results/video_frame_prompt\
#   --agent_type video_prompt\
#   --video_dir media\
#   --max_frame_num 60