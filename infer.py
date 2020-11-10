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

from dataset import InferDataset
from model import ResNet50
from ontology_reader import OntologyReader
from utils import read_jsonl, top_k_accuracy, jaccard_similarity, cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Event classification inference script")

    # Common arguments
    parser.add_argument("-v", "--debug", action="store_true", help="debug output")

    # Required arguments
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to yml cfg")
    parser.add_argument("-i", "--images", type=str, nargs="+", required=True, help="Root image or tf_record dir")

    # Optional arguments
    parser.add_argument("-b", "--batch_size", type=str, required=False, default=32, help="Batch size")
    parser.add_argument("--num_predictions", type=int, required=False, default=3, help="Number of predictions to show")
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

                if prediction is None:  # function returned error
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
                "leaf_node_vector": leaf_node_vector,
                "subgraph_vector": subgraph_vector
            }

    return sample_predictions


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
    infer_dataset = InferDataset(image_paths=args.images)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args.batch_size, num_workers=8)

    # predict event classes for images
    sample_predictions = get_sample_predictions(infer_dataloader=infer_dataloader,
                                                OntReader=OntReader,
                                                model=model,
                                                device=device,
                                                s2l_strategy=args.s2l_strategy)

    for sample, values in sample_predictions.items():
        leaf_node_vector = values["leaf_node_vector"]
        prediction = sorted(range(len(leaf_node_vector)), key=lambda k: -leaf_node_vector[k].item())

        logging.info("################")
        logging.info(f"Prediction for: {sample}")
        for i in range(3):
            predicted_class = OntReader.node_from_classidx[prediction[i]]
            prob = leaf_node_vector[predicted_class["class_idx"]]
            prob = "{:0.4f}".format(prob)
            logging.info(
                f"=> Top {i+1} (prob: {prob}): {predicted_class['wd_label']} (wikidata id: {predicted_class['wd_id']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
