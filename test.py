import argparse
import json
import logging
import numpy as np
import os
import pickle
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import yaml

from dataset import EventDataset
from model import ResNet50
from ontology_reader import OntologyReader
from utils import read_jsonl, top_k_accuracy, jaccard_similarity, cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Event classification evaluation script")

    # Common arguments
    parser.add_argument("-v", "--debug", action="store_true", help="enable debug outputs")

    # required arguments
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to config yaml file of the model")
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("-t", "--testset", required=True, help="Path to <testset>.jsonl")

    # optional arguments
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to output file [json or jsonl]")
    parser.add_argument("--batch_size", type=int, required=False, default=16, help="Batch size")
    parser.add_argument("--s2l_strategy",
                        type=str,
                        required=False,
                        default="leafprob*cossim",
                        choices=["leafprob", "cossim", "leafprob*cossim"],
                        help="strategy to convert the subgraph vectors to leaf node vectors")

    args = parser.parse_args()
    return args


def get_sample_predictions(infer_dataloader, OntReader, model, device, s2l_strategy):
    sample_predictions = {}

    for batch in tqdm(infer_dataloader, desc="Get predictions for images ..."):  # loop over batches
        batch_result = model(batch["image"].to(device))
        for sample in range(batch_result["predictions"].shape[0]):  # loop over samples in batches

            # get prediction from model
            prediction = batch_result["predictions"][sample, :].detach().cpu().numpy()
            if model.model_type == "classification":
                leaf_node_vector = prediction
            elif model.model_type == "ontology":
                # convert predicted subgraph vector to leaf node vector
                leaf_node_vector = OntReader.subgraph_to_leaf_vector(pred_subgraph_vector=prediction,
                                                                     strategy=s2l_strategy,
                                                                     redundancy_removal=model.redundancy_removal)

                if leaf_node_vector is None:  # function returned error
                    logging.error(
                        "Conversion from subgraph vector to leaf node vector failed! Correct config parameters?")
                    return {}
            else:
                logging.error("Unknown model type in cfg! Please use [classification, ontology]!")
                return {}

            # get multi-hot encoded subgraph vector from predicted leaf node vector
            subgraph_vector = OntReader.leaf_to_subgraph_vector(leaf_node_vector)

            # store sample prediction
            sample_predictions[batch["image_path"][sample]] = {
                "image_path": batch["image_path"][sample],
                "gt_leaf_class_idx": batch["leaf_class_idx"][sample].item(),
                "gt_leaf_wd_id": batch["leaf_wd_id"][sample],
                "leaf_node_vector": leaf_node_vector,
                "subgraph_vector": subgraph_vector
            }

    return sample_predictions


def get_test_results(sample_predictions, OntReader):
    node_results = {}

    for sample in sample_predictions.values():
        # get sample ground truth
        gt_class_idx = sample["gt_leaf_class_idx"]
        gt_subgraph_vector = OntReader.get_subgraph_vector(sample["gt_leaf_wd_id"])
        gt_subgraph_nodes = OntReader.get_subgraph_nodes(sample["gt_leaf_wd_id"])

        # get sample prediction
        pred_leaf_node_vector = sample["leaf_node_vector"]
        pred_subgraph_vector = sample["subgraph_vector"]

        # calculate metrics
        accuracy = top_k_accuracy(gt_class_idx, pred_leaf_node_vector, kvals=[1, 3, 5])
        jaccard = jaccard_similarity(gt_subgraph_vector, pred_subgraph_vector)
        cosine = cosine_similarity(gt_subgraph_vector, pred_subgraph_vector)

        # set results for each node in the subgraph of the gt leaf event node
        for node in gt_subgraph_nodes:
            if node["wd_id"] not in node_results:
                node_results[node["wd_id"]] = {
                    "wd_id": node["wd_id"],
                    "wd_label": node["wd_label"],
                    "num_test_images": 0,
                    "metrics": {
                        "accuracy-top1": 0,
                        "accuracy-top3": 0,
                        "accuracy-top5": 0,
                        "jaccard": 0,
                        "cosine": 0,
                    },
                }

            node_results[node["wd_id"]]["num_test_images"] += 1
            node_results[node["wd_id"]]["metrics"]["accuracy-top1"] += accuracy[0]
            node_results[node["wd_id"]]["metrics"]["accuracy-top3"] += accuracy[1]
            node_results[node["wd_id"]]["metrics"]["accuracy-top5"] += accuracy[2]
            node_results[node["wd_id"]]["metrics"]["jaccard"] += jaccard
            node_results[node["wd_id"]]["metrics"]["cosine"] += cosine

    return node_results


def print_results(metrics, images):
    for metric, result in metrics.items():
        logging.info(f"{metric}: {(100 * result / images):.1f}")


def main():
    args = parse_args()
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    # load cfg
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            logging.debug(cfg)
    else:
        logging.error(f"Cannot find cfg file: {args.cfg}")
        return 0

    # load ontology
    OntReader = OntologyReader(graph_file=os.path.join(os.path.dirname(args.cfg), cfg["graph"]),
                               weighting_scheme=cfg["weighting_scheme"],
                               leaf_node_weight=cfg["leaf_node_weight"])

    # init torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        batch_size = torch.cuda.device_count() * args.batch_size
    else:
        batch_size = args.batch_size

    # build model and load checkpoint
    if cfg["model_type"] == "ontology":
        weights = OntReader.get_node_weights(cfg["redundancy_removal"])
        num_classes = len(weights)
    else:  # cfg["model_type"] == "classification"
        num_classes = OntReader.num_leafs

    if torch.cuda.device_count() == 0:
        logging.info(f"Test on CPU with batch_size {batch_size}")
    else:
        logging.info(f"Test on {torch.cuda.device_count()} GPU(s) with batch_size {batch_size}")

    model = ResNet50(num_classes=num_classes,
                     model_type=cfg["model_type"],
                     redundancy_removal=cfg["redundancy_removal"])
    model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Found {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.eval()
    model.load(device=device, path=os.path.join(os.path.dirname(args.cfg), cfg["model_checkpoint"]))

    # Init testing dataset
    infer_dataset = EventDataset(image_dir=args.image_dir, testset_path=args.testset)
    infer_dataloader = DataLoader(infer_dataset, batch_size=batch_size, num_workers=8)

    # predict event classes for images
    sample_predictions = get_sample_predictions(infer_dataloader=infer_dataloader,
                                                OntReader=OntReader,
                                                model=model,
                                                device=device,
                                                s2l_strategy=args.s2l_strategy)

    # calculate result for all nodes in the ontology
    logging.info("Calculate results ...")
    node_results = get_test_results(sample_predictions=sample_predictions, OntReader=OntReader)

    # print final results (global results are stored in the root node occurrence (Q1190554))
    if "Q1190554" not in node_results:
        logging.warning("No results written ...")
        return 0

    print_results(node_results["Q1190554"]["metrics"], node_results["Q1190554"]["num_test_images"])

    # write results for each node
    if args.output:
        if not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

        result_list = []
        for val in node_results.values():
            # calculate mean result
            for metric, result in val["metrics"].items():
                val["metrics"][metric] = result / val["num_test_images"]
            result_list.append(val)

        result_list = sorted(result_list, key=lambda x: x["num_test_images"], reverse=True)
        with open(args.output, "w") as jsonfile:
            for result in result_list:
                jsonfile.write(json.dumps(result) + "\n")

        logging.info(f"Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
