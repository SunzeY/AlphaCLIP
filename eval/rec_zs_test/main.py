from collections import defaultdict
import json
import argparse
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from interpreter import *
from executor import *
from methods import *

METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, help="path to images (train2014 directory of COCO)")
    parser.add_argument("--clip_model", type=str, default="RN50x16,ViT-B/32", help="which clip model to use (should use RN50x4, ViT-B/32, or both separated by a comma")
    parser.add_argument("--clip_type", type=str, default="aclip", help="which clip model to use (should use RN50x4, ViT-B/32, or both separated by a comma")
    parser.add_argument("--albef_path", type=str, default=None, help="to use ALBEF (instead of CLIP), specify the path to the ALBEF checkpoint")
    parser.add_argument("--method", type=str, default="parse", help="method to solve expressions")
    parser.add_argument("--box_representation_method", type=str, default="crop,blur", help="method of representing boxes as individual images (crop, blur, or both separated by a comma)")
    parser.add_argument("--box_method_aggregator", type=str, default="sum", help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.0, help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--output_file", type=str, default=None, help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default=None, help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--mock", action="store_true", help="(optional) mock CLIP execution.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    parser.add_argument("--shuffle_words", action="store_true", help="If true, shuffle words in the sentence")
    parser.add_argument("--gradcam_alpha", type=float, nargs='+', help="alpha value to use for gradcam method")
    parser.add_argument("--enlarge_boxes", type=float, default=0.0, help="(optional) whether to enlarge boxes when passing them to the model")
    parser.add_argument("--part", type=str, default=None, help="(optional) specify how many parts to divide the dataset into and which part to run in the format NUM_PARTS,PART_NUM")
    parser.add_argument("--batch_size", type=int, default=1, help="number of instances to process in one model call (only supported for baseline model)")
    parser.add_argument("--baseline_head", action="store_true", help="For baseline, controls whether model is called on both full expression and head noun chunk of expression")
    parser.add_argument("--mdetr", type=str, default=None, help="to use MDETR as the executor model, specify the name of the MDETR model")
    parser.add_argument("--albef_block_num", type=int, default=8, help="block num for ALBEF gradcam")
    parser.add_argument("--albef_mode", type=str, choices=["itm", "itc"], default="itm")
    parser.add_argument("--expand_position_embedding",action="store_true")
    parser.add_argument("--gradcam_background", action="store_true")
    parser.add_argument("--mdetr_given_bboxes", action="store_true")
    parser.add_argument("--mdetr_use_token_mapping", action="store_true")
    parser.add_argument("--non_square_size", action="store_true")
    parser.add_argument("--blur_std_dev", type=int, default=100, help="standard deviation of Gaussian blur")
    parser.add_argument("--gradcam_ensemble_before", action="store_true", help="Average gradcam maps of different models before summing over the maps")
    parser.add_argument("--cache_path", type=str, default=None, help="cache features")
    # Arguments related to Parse method.
    parser.add_argument("--no_rel", action="store_true", help="Disable relation extraction.")
    parser.add_argument("--no_sup", action="store_true", help="Disable superlative extraction.")
    parser.add_argument("--no_null", action="store_true", help="Disable null keyword heuristics.")
    parser.add_argument("--ternary", action="store_true", help="Disable ternary relation extraction.")
    parser.add_argument("--baseline_threshold", type=float, default=float("inf"), help="(Parse) Threshold to use relations/superlatives.")
    parser.add_argument("--temperature", type=float, default=1., help="(Parse) Sigmoid temperature.")
    parser.add_argument("--superlative_head_only", action="store_true", help="(Parse) Superlatives only quanntify head predicate.")
    parser.add_argument("--sigmoid", action="store_true", help="(Parse) Use sigmoid, not softmax.")
    parser.add_argument("--no_possessive", action="store_true", help="(Parse) Model extraneous relations as possessive relations.")
    parser.add_argument("--expand_chunks", action="store_true", help="(Parse) Expand noun chunks to include descendant tokens that aren't ancestors of tokens in other chunks")
    parser.add_argument("--parse_no_branch", action="store_true", help="(Parse) Only do the parsing procedure if some relation/superlative keyword is in the expression")
    parser.add_argument("--possessive_no_expand", action="store_true", help="(Parse) Expand ent2 in possessive case")
    args = parser.parse_args()

    with open(args.input_file) as f: 
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    gradcam = args.method == "gradcam"

    executor = ClipExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method, method_aggregator=args.box_method_aggregator, device=device, square_size=not args.non_square_size, expand_position_embedding=args.expand_position_embedding, blur_std_dev=args.blur_std_dev, cache_path=args.cache_path, input_file=args.input_file, clip_type=args.clip_type)

    method = METHODS_MAP[args.method](args)
    correct_count = 0
    total_count = 0
    if args.output_file:
        output_file = open(args.output_file, "w")
    if args.detector_file:
        detector_file = open(args.detector_file)
        detections_list = json.load(detector_file)
        if isinstance(detections_list, dict):
            detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
        else:
            detections_map = defaultdict(list)
            for detection in detections_list:
                detections_map[detection["image_id"]].append(detection["box"])
    
    part = 0
    if args.part is not None: # for multi-gpu test / part-data test
        num_parts = int(args.part.split(",")[0])
        part = int(args.part.split(",")[1])
        data = data[int(len(data)*part/num_parts):int(len(data)*(part+1)/num_parts)]

    batch_count = 0
    batch_boxes = []
    batch_gold_boxes = []
    batch_gold_index = []
    batch_file_names = []
    batch_sentences = []
    for datum in tqdm(data):
        if "coco" in datum["file_name"].lower():
            file_name = "_".join(datum["file_name"].split("_")[:-1])+".jpg"
        else:
            file_name = datum["file_name"]
        img_path = os.path.join(args.image_root, file_name)
        img = Image.open(img_path).convert('RGB')
        gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in datum["anns"]]
        if isinstance(datum["ann_id"], int) or isinstance(datum["ann_id"], str):
            datum["ann_id"] = [datum["ann_id"]]
        assert isinstance(datum["ann_id"], list)
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]] 
        if args.detector_file:
                boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]
                if len(boxes) == 0:
                    boxes = [Box(x=0, y=0, w=img.width, h=img.height)]
        else:
            boxes = gold_boxes
        for sentence in datum["sentences"]:
            env = Environment(img, boxes, executor, (args.mdetr is not None and not args.mdetr_given_bboxes), str(datum["image_id"]), img_path) 
            if args.shuffle_words:
                words = sentence["raw"].lower().split()
                random.shuffle(words)
                result = method.execute(" ".join(words), env)
            else:
                result = method.execute(sentence["raw"].lower(), env)
            boxes = env.boxes
            print(sentence["raw"].lower())
            correct = False
            for g_index in gold_index:
                if iou(boxes[result["pred"]], gold_boxes[g_index]) > 0.5:
                    correct = True
                    break
            if correct:
                result["correct"] = 1
                correct_count += 1
            else:
                result["correct"] = 0
            if args.detector_file:
                argmax_ious = []
                max_ious = []
                for g_index in gold_index:
                    ious = [iou(box, gold_boxes[g_index]) for box in boxes]
                    argmax_iou = -1
                    max_iou = 0
                    if max(ious) >= 0.5:
                        for index, value in enumerate(ious):
                            if value > max_iou:
                                max_iou = value
                                argmax_iou = index
                    argmax_ious.append(argmax_iou)
                    max_ious.append(max_iou)
                argmax_iou = -1
                max_iou = 0
                if max(max_ious) >= 0.5:
                    for index, value in zip(argmax_ious, max_ious):
                        if value > max_iou:
                            max_iou = value
                            argmax_iou = index
                result["gold_index"] = argmax_iou
            else:
                result["gold_index"] = gold_index
            result["bboxes"] = [[box.left, box.top, box.right, box.bottom] for box in boxes]
            result["file_name"] = file_name
            result["probabilities"] = result["probs"]
            result["text"] = sentence["raw"].lower()
            if args.output_file:
                # Serialize numpy arrays for JSON.
                for key in result:
                    if isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()
                    if isinstance(result[key], np.int64):
                        result[key] = result[key].item()
                output_file.write(json.dumps(result)+"\n")
            total_count += 1
            print(f"est_acc: {100 * correct_count / total_count:.3f}")

    if args.output_file:
        output_file.close()
    print(f"acc: {100 * correct_count / total_count:.3f}")
    acc = 100 * correct_count / total_count
    
    result = {}
    result['acc'] = acc
    json.dump(acc, open(os.path.join('./output', args.input_file.split('/')[-1].split('.')[0] + '_acc_' + str(part)+'.json'),'w'))
    json.dump(str(correct_count)+' '+str(total_count), open(os.path.join('./output', args.input_file.split('/')[-1].split('.')[0] + '_count_' + str(part)+'.json'),'w'))
    stats = method.get_stats()
    if stats:
        pairs = sorted(list(stats.items()), key=lambda tup: tup[0])
        for key, value in pairs:
            result[key] = value
            if isinstance(value, float):
                print(f"{key}: {value:.5f}")
            else:
                print(f"{key}: {value}")

    json.dump(result, open(os.path.join('./output', args.input_file.split('/')[-1].split('.')[0] + '_' + str(part)+'.json'),'w'))