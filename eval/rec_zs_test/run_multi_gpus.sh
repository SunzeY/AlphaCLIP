CUDA_VISIBLE_DEVICES=0 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,0" &

CUDA_VISIBLE_DEVICES=1 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,1" &

CUDA_VISIBLE_DEVICES=2 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,2" &

CUDA_VISIBLE_DEVICES=3 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,3" &

CUDA_VISIBLE_DEVICES=4 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,4" &

CUDA_VISIBLE_DEVICES=5 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,5" &

CUDA_VISIBLE_DEVICES=6 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,6" &

CUDA_VISIBLE_DEVICES=7 python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --box_representation_method full,blur --detector_file reclip_data/refcoco_dets_dict.json --cache_path ./cache --part "8,7" 