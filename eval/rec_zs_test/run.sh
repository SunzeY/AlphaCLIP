CUDA_VISIBLE_DEVICES=0 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_representation_method full,blur --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache