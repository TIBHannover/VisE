import logging
import networkx as nx
import numpy as np
import sys

from utils import read_jsonl, cosine_similarity


class OntologyReader:
    def __init__(self, graph_file, weighting_scheme=None, leaf_node_weight=1.0):

        logging.info("Reading Ontology ...")
        self._graph = nx.read_graphml(graph_file)

        self.nodes = {}
        self.node_from_classidx = {}
        self.num_leafs = 0
        for node in self._graph.nodes(data=True):
            node_wd_id = node[0]
            self.nodes[node_wd_id] = node[1]

            if node[1]["node_type"] == "leaf":
                self.num_leafs += 1
                self.node_from_classidx[node[1]["class_idx"]] = node[1]

        logging.info(nx.info(self._graph))
        logging.info(f"Number of Leaf Event Nodes: {self.num_leafs}")
        logging.info(f"Number of Branch Event Nodes: {len(self.nodes.keys()) - self.num_leafs}")

        logging.info("Build Subgraphs ...")
        self.subgraphs = {}
        for node in self.nodes.values():
            if node["node_type"] != "leaf":
                continue

            subgraph_vector, subgraph_nodes = self._get_subgraph_information(node["wd_id"], idx="ontology_idx")
            subgraph_RR_vector, subgraph_RR_nodes = self._get_subgraph_information(node["wd_id"], idx="ontology_RR_idx")

            self.subgraphs[node["wd_id"]] = {
                "wd_id": node["wd_id"],
                "class_idx": node["class_idx"],
                "subgraph_vector": subgraph_vector,
                "subgraph_nodes": subgraph_nodes,
                "ontology_idx": node["ontology_idx"],
                "subgraph_RR_vector": subgraph_RR_vector,
                "subgraph_RR_nodes": subgraph_RR_nodes,
                "ontology_RR_idx": node["ontology_RR_idx"]
            }

        logging.info("Calculate Node Weights ...")
        self._node_weights = self._calc_node_weights(weighting_scheme, leaf_node_weight, idx="ontology_idx")
        self._node_RR_weights = self._calc_node_weights(weighting_scheme, leaf_node_weight, idx="ontology_RR_idx")

    def leaf_to_subgraph_vector(self, leaf_node_vector, redundancy_removal=False):
        # get index of max prediction and corresponding wd_id
        predicted_class_idx = np.argmax(leaf_node_vector)
        predicted_wd_id = self.node_from_classidx[predicted_class_idx]["wd_id"]
        return self.get_subgraph_vector(predicted_wd_id, redundancy_removal=redundancy_removal)

    def subgraph_to_leaf_vector(self, pred_subgraph_vector, strategy, redundancy_removal=False):

        weights = self.get_node_weights(redundancy_removal=redundancy_removal)
        leaf_node_vec = np.zeros(shape=[self.num_leafs])
        subgraph_mat = np.zeros(shape=[self.num_leafs, len(weights)])

        for subgraph in self.subgraphs.values():
            subgraph_idx = self.get_subgraph_vector_index(subgraph["wd_id"], redundancy_removal=redundancy_removal)
            subgraph_vector = self.get_subgraph_vector(subgraph["wd_id"], redundancy_removal=redundancy_removal)

            leaf_node_vec[subgraph["class_idx"]] = pred_subgraph_vector[subgraph_idx]
            subgraph_mat[subgraph["class_idx"], :] = subgraph_vector

        if strategy == "leafprob":
            return leaf_node_vec

        if "cossim" in strategy:
            cos_sim = cosine_similarity(gt=subgraph_mat, prediction=pred_subgraph_vector, weights=weights)

            if strategy == "cossim":
                return cos_sim
            elif strategy == "leafprob*cossim":
                return leaf_node_vec * cos_sim

        logging.error("Unknown ont2cls strategy. Exiting!")
        return None

    def get_leaf_node_vector(self, wd_id):
        leaf_node_vector = [0.0] * self.num_leafs
        node_idx = self.nodes[wd_id]["class_idx"]
        leaf_node_vector[node_idx] = 1.0

        return leaf_node_vector

    def get_class_idx(self, wd_id):
        return self.nodes[wd_id]["class_idx"]

    def get_subgraph_vector(self, wd_id, redundancy_removal=False):
        if redundancy_removal:
            return self.subgraphs[wd_id]["subgraph_RR_vector"]
        else:
            return self.subgraphs[wd_id]["subgraph_vector"]

    def get_subgraph_nodes(self, wd_id, redundancy_removal=False):
        if redundancy_removal:
            return self.subgraphs[wd_id]["subgraph_RR_nodes"]
        else:
            return self.subgraphs[wd_id]["subgraph_nodes"]

    def get_subgraph_vector_index(self, wd_id, redundancy_removal=False):
        if redundancy_removal:
            return self.subgraphs[wd_id]["ontology_RR_idx"]
        else:
            return self.subgraphs[wd_id]["ontology_idx"]

    def get_node_weights(self, redundancy_removal=False):
        if redundancy_removal:
            return self._node_RR_weights
        else:
            return self._node_weights

    def _get_subgraph_information(self, wd_id, idx):
        # get number of nodes in the subgraph vector
        num_nodes = 0
        for node in self.nodes.values():
            if node[idx] != -1:  # if redundant nodes are removed
                num_nodes += 1

        subgraph_vector = [0.0] * num_nodes
        subgraph_nodes = []
        preorder_nodes = list(nx.dfs_preorder_nodes(self._graph, source=wd_id))

        for node in preorder_nodes:
            if self.nodes[node][idx] == -1:  # if redundant nodes are removed
                continue

            node_idx = self.nodes[node][idx]
            subgraph_vector[node_idx] = 1.0
            subgraph_nodes.append({"wd_id": self.nodes[node]["wd_id"], "wd_label": self.nodes[node]["wd_label"]})

        return np.asarray(subgraph_vector), subgraph_nodes

    def _calc_node_weights(self, weighting_scheme, leaf_node_weight, idx):
        # get number of nodes in the subgraph vector
        num_nodes = 0
        for node in self.nodes.values():
            if node[idx] != -1:
                num_nodes += 1

        node_weights = [1.0] * num_nodes

        if weighting_scheme is None:
            return node_weights

        leafs = set()
        for node in self.nodes.values():

            if node["node_type"] == "leaf":
                leafs.add(node[idx])
                continue

            if weighting_scheme == "distance":
                node_weights[node[idx]] = self._get_distance_weight(node["wd_id"])

            elif weighting_scheme == "centrality":
                node_weights[node[idx]] = self._get_degree_of_centrality_weight(node["wd_id"])

            else:
                logging.error("Unknown weight type. Exiting ...")
                exit()

        for leaf_idx in leafs:
            node_weights[leaf_idx] = leaf_node_weight

        return node_weights

    def _get_distance_weight(self, wd_id):
        predecessors = nx.bfs_tree(self._graph, wd_id, reverse=True)
        sum_leaf_distance = 0
        connected_leafs = 0
        for p in predecessors:  # determine the shortest path to each related leaf event node
            if p not in self.nodes:  # if redundant nodes are removed
                continue

            if self.nodes[p]["node_type"] == "leaf":
                connected_leafs += 1
                shortest_path = nx.shortest_path(self._graph, source=self.nodes[p]["wd_id"], target=wd_id)
                sum_leaf_distance += len(shortest_path)

        avg_leaf_distance = sum_leaf_distance / connected_leafs
        return 1 / (2**(avg_leaf_distance - 1))  # Eq. (4 of the paper)

    def _get_degree_of_centrality_weight(self, wd_id):
        predecessors = nx.bfs_tree(self._graph, wd_id, reverse=True)
        connected_leafs = 0

        for p in predecessors:  # determine the number of leaf event nodes connected
            if p not in self.nodes:  # if redundant nodes are removed
                continue

            if self.nodes[p]["node_type"] == "leaf":
                connected_leafs += 1

        return 1 - (connected_leafs - 1) / self.num_leafs  # Eq. (5) of the paper
