# classification (aid + ucmerced )
python projects/llava_sam2/evaluation/classify.py --model_path train_finalnew --results_dir aid_result --annotation_file data/GeoChat-Bench/aid.jsonl --image_dir data/aid_test_image
python projects/llava_sam2/evaluation/classify.py --model_path train_finalnew --results_dir uic_result --annotation_file data/GeoChat-Bench/UCmerced.jsonl --image_dir data/UCMerced_LandUse/Images

# HRBEN VQA
python projects/llava_sam2/evaluation/infer_hb.py --model_path train_finalnew --results_dir hr_vqaresult --annotation_file data/GeoChat-Bench/hrben.jsonl --image_dir data/HRBEN
python projects/llava_sam2/evaluation/evaluate_hrben.py

# VRSBench VQA
python projects/llava_sam2/evaluation/infer_vrsbench_vqa.py --model_path train_finalnew --results_dir vrs_vqaresult --annotation_file data/vrsbenchvqa/VRSBench_EVAL_vqa.json --image_dir data/vrsbenchvqa/Images_val
python projects/llava_sam2/evaluation/evaluate_vrsbench_vqa.py


#VRSBench Cap
python projects/llava_sam2/evaluation/infer_vrsbench_cap.py --model_path train_finalnew --results_dir vrs_capresult --annotation_file data/vrsbenchvqa/VRSBench_EVAL_Cap.json --image_dir data/vrsbenchvqa/Images_val
python projects/llava_sam2/evaluation/caption_eval.py
python projects/llava_sam2/evaluation/create_json_references.py -i gt_cap.txt -o gt_cap.json
python projects/llava_sam2/evaluation/run_evaluations.py -i pred_cap.txt -r gt_cap.json








