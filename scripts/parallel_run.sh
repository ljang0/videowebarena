#!/bin/bash  
#   input_texts=("test_classifieds" "test_gitlab" "test_map" "test_reddit" "test_shopping" "test_shopping_admin")  

# List of input texts  
input_texts=("test_classifieds" "test_gitlab" "test_map" "test_reddit" "test_shopping" "test_shopping_admin")  
  
# Iterate over each input text  
for input_text in "${input_texts[@]}"; do  
  # Extract the substring starting from the 6th character  
  session_name="${input_text:5}"  
    
  # Create the first screen session and run the script  
  # screen -dmS "$session_name" bash -c ". scripts/run_video.sh $input_text; exit"  
    
  # Create the second screen session with "F" appended and run the script  
  screen -dmS "${session_name}" bash -c ". scripts/run_video_frame_summary.sh $input_text; exit"  
done  
