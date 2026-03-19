# evaluate mcq tasks
python projects/llava_sam2/evaluation/infer_earthmindbench_mcq.py --model_path train_finalnew --results_dir pair_multimcq --annotation_files data/formal_data_test/hallucination_detection_all_unmatched.json data/formal_data_test/object_counting_all_unmatched.json data/formal_data_test/object_existence_all_unmatched.json data/formal_data_test/scene_classification_unmatched.json data/formal_data_test/spatial_relationship_all_unmatched.json --image_dir data/pair_data/test
python projects/llava_sam2/evaluation/evaluate_earthmindbench_mcq.py

# evaluate oe tasks
python projects/llava_sam2/evaluation/infer_earthmindbench_oe.py --model_path train_finalnew --results_dir pair_multioe --annotation_files data/formal_data_test/caption_all_unmatched.json data/formal_data_test/disaster_all_unmatched.json data/formal_data_test/Navigation_all_unmatched.json data/formal_data_test/urban_all_unmatched.json data/formal_data_test/spatial_relationship_all_unmatched.json --image_dir data/pair_data/test
python projects/llava_sam2/evaluation/evaluate_earthmindbench_oe.py

# evaluate segmentatioin tasks
python projects/llava_sam2/evaluation/refrs_eval_multi.py --model_path train_finalnew --dataset sarnew # earthmind-bench-seg