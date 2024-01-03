# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

import regex as re
import torch
from transformers import BatchEncoding

from preprocessing import usg, datasets



class Document:
    @staticmethod
    def parse_from(obj: Document):
        if obj is None:
            return None
        references = {}
        doc = Document(str(obj.text))
        references[obj] = doc
        doc.sentences = [Sentence.parse_from(s, references) for s in obj.sentences] if obj.sentences else []
        doc.unified_semantic_graph = usg.Graph.parse_from(obj.unified_semantic_graph, references)
        doc.entity_clusters = [EntityCluster.parse_from(s, references) for s in
                               obj.entity_clusters] if obj.entity_clusters else []

        # i = 0
        # for si, sentence in enumerate(doc.sentences):
        #     doc.tokens.extend(sentence.tokens)
        #     for token in sentence.tokens:
        #         token.index_in_document = i
        #         token.sentence_index = si
        #         i += 1
        return doc

    def __init__(self, text: str):
        self.sentences: list[Sentence] = []
        self.entity_clusters: list[EntityCluster] = []
        self.unified_semantic_graph: usg.Graph = None
        self.text: str = text
        self.tokens: list[Token] = []

    # this method implements the pseudo code in the appendix of the bass paper.
    # for clarification, we also consulted the provided source code,
    # but we let details in the paper take precedence over implementation details in the source code.
    # this method assumes that the annotation output of CoreNLP has already been parsed, this includes:
    #   - coreference resolution
    #   - joining coreference clusters from multiple annotation outputs into a single collection
    #   - parsing dependency trees of sentences as graphs
    #       - with one node per token
    #       - with nodes being connected as per the dependency relation of their tokens
    #       - identifying node types (POS tag) from the associated token
    def build_strictly_compliant_with_bass_paper(self, with_graph_augmentations=True):
        i = 0
        for si, sentence in enumerate(self.sentences):
            self.tokens.extend(sentence.tokens)
            for token in sentence.tokens:
                token.index_in_document = i
                token.sentence_index = si
                i += 1

        # concatenate all dependency trees into a single graph for future merging operations
        # the paper does this later, but it does not matter
        self.unified_semantic_graph = usg.Graph()
        for sentence in self.sentences:
            self.unified_semantic_graph.nodes.extend(sentence.semantic_graph.nodes)
            self.unified_semantic_graph.edges.extend(sentence.semantic_graph.edges)
            self.unified_semantic_graph.roots.extend(sentence.semantic_graph.roots)

        for i, node in enumerate(self.unified_semantic_graph.nodes):
            node.index = i

        # REMOVE_PUNCTUATION
        # remove all punctuation nodes and their edges
        for node in self.unified_semantic_graph.nodes[:]:
            if node.root.dependency_relation == "punct" or node.root.xpos.startswith((".", ",", "?", "!", ":", "(", ")")):
                self.unified_semantic_graph.remove_node(node)

        # MERGE_COREF_PHRASE
        # merge all nodes of mentions into a single phrase
        for cluster in self.entity_clusters:
            for entity in cluster.mentions[:]:
                for token in entity.tokens:
                    self.unified_semantic_graph.merge_nodes(entity.root.node, token.node)

        # MERGE_NODES
        # apply linguistic merging rules depth-first
        for root in self.unified_semantic_graph.roots:
            self.recursively_collapse_nodes_to_semantic_graph(self.unified_semantic_graph, root)

        # MERGE_PHRASE
        # merge phrases of coreference chains
        for cluster in self.entity_clusters:
            for entity in cluster.mentions[:]:
                self.unified_semantic_graph.merge_nodes(cluster.representative.root.node, entity.root.node, replace_tokens=True)

        # and equal phrases (equal by terms of string equality)
        phrases = {}
        for node in self.unified_semantic_graph.nodes:
            n = phrases.get(str(node))
            if not n:
                n = node
                phrases[str(node)] = node

            self.unified_semantic_graph.merge_nodes(n, node, replace_tokens=True)

        if with_graph_augmentations:
            self.create_two_hop_connections()
            self.create_supernode()
            
            self.unified_semantic_graph.nodes = self.unified_semantic_graph.nodes[-1:] + self.unified_semantic_graph.nodes[0:-1]

        for i, node in enumerate(self.unified_semantic_graph.nodes):
            node.index = i


    def create_supernode(self):
        node_index = len(self.unified_semantic_graph.nodes)
        supernode = usg.Node()
        supernode.index = node_index
        supernode.root = Token()
        for node in self.unified_semantic_graph.nodes:
            self.unified_semantic_graph.add_edge(supernode, node, "SUPERNODE")

        self.unified_semantic_graph.nodes.append(supernode)

    def recursively_collapse_nodes_to_semantic_graph(self, semantic_graph: usg.Graph, node: usg.Node,
                                                     visited_nodes: list[usg.Node] = []):
        # the original source code recursively iterates at the end of this method (before the call to remove_case_children),
        # effectively doing a breadth search, however, the original paper states doing a depth search instead
        visited_nodes.append(node)
        edges_with_unvisited_destination = [edge for edge in node.outgoing_edges if
                                            edge.destination not in visited_nodes]
        while edges_with_unvisited_destination:
            edge = edges_with_unvisited_destination[0]
            visited_nodes.append(edge.destination)
            self.recursively_collapse_nodes_to_semantic_graph(semantic_graph, edge.destination, visited_nodes)

            edges_with_unvisited_destination = [edge for edge in node.outgoing_edges if
                                                edge.destination not in visited_nodes]

        if node not in self.unified_semantic_graph.nodes:
            return

        if node.root.xpos.startswith("VB"):
            if node.root.dependency_relation == "cop":
                self.collapse_copula_verb(semantic_graph, node)
            else:
                self.collapse_verb(semantic_graph, node)
        if node.root.xpos.startswith(("NN", "JJ")):
            self.collapse_noun_phrase(semantic_graph, node)
        elif node.root.xpos.startswith("CD"):
            self.collapse_cd_phrase(semantic_graph, node)
        elif self.is_punctuation(node):
            semantic_graph.remove_node(node)
        else:
            self.collapse_fixed_compounds(semantic_graph, node)

        self.remove_case_children(semantic_graph, node)

    @staticmethod
    def remove_case_children(graph: usg.Graph, node: usg.Node):
        for edge in node.outgoing_edges[:]:
            if (edge.relation.startswith(
                    ("case", "aux", "mark")) or edge.relation == "cc") and not edge.destination.outgoing_edges:
                graph.remove_node(edge.destination)

    @staticmethod
    def collapse_fixed_compounds(semantic_graph: usg.Graph, node: usg.Node):
        for edge in node.outgoing_edges[:]:
            if edge.relation.startswith("mwe"):
                semantic_graph.merge_nodes(node, edge.destination.root.node, transfer_edges=False)


    @staticmethod
    def collapse_cd_phrase(semantic_graph: usg.Graph, node: usg.Node):
        for edge in node.outgoing_edges[:]:
            if edge.relation.startswith("mwe"):
                semantic_graph.merge_nodes(node, edge.destination.root.node)

    @staticmethod
    def collapse_noun_phrase(semantic_graph: usg.Graph, node: usg.Node):
        for edge in node.outgoing_edges[:]:
            if edge.relation.startswith(("compound", "name", "amod", "nummod", "predet", "advmod")) or edge.relation == "det":
                semantic_graph.merge_nodes(node, edge.destination.root.node)


    @staticmethod
    def collapse_verb(semantic_graph: usg.Graph, node: usg.Node):
        for edge in node.outgoing_edges[:]:
            if edge.relation in ("compound", "aux") or edge.relation.startswith(
                    ("auxpass", "neg", "expl", "advmod")):
                semantic_graph.merge_nodes(node, edge.destination.root.node)


    @staticmethod
    def collapse_copula_verb(semantic_graph: usg.Graph, node: usg.Node):
        parent_node = node.root.dependency_head.node
        if parent_node is None:
            print("parent_node is None")
            parent_node = node.incoming_edges[0].source

        # swap node with parent
        # then collapse node as if it were a verb

        for edge in parent_node.outgoing_edges[:]:
            if edge.relation.startswith("nsubj"):
                semantic_graph.change_edge_source(edge, node)
            if edge.relation.startswith(("mwe", "aux", "case", "mark", "neg")):
                if edge.destination.root.index_in_sentence > node.root.index_in_sentence:
                    semantic_graph.remove_edge(edge)
                    semantic_graph.merge_nodes(node, edge.destination.root.node)

        for edge in parent_node.incoming_edges[:]:
            if edge.relation.startswith(("ccomp", "xcomp", "ROOT")):
                semantic_graph.change_edge_destination(edge, node)


    def export_graph_information_for_tokenization(self, tokenized_text: BatchEncoding):
        self.identify_token_ids(tokenized_text)
        graph_construction_matrix = torch.zeros(
            (len(self.unified_semantic_graph.nodes), tokenized_text.input_ids.size(1)), dtype=torch.bool)

        for node in self.unified_semantic_graph.nodes:
            tokens = node.tokens[:]
            for ts in node.replaced_tokens:
                tokens.extend(ts)
            tokens = list(set(tokens))
            for token in tokens:
                for id in token.token_ids:
                    graph_construction_matrix[node.index, id] = True

        adjacency_matrix = torch.eye(len(self.unified_semantic_graph.nodes), dtype=torch.bool)
        for node in self.unified_semantic_graph.nodes:
            for edge in node.outgoing_edges:
                adjacency_matrix[node.index, edge.destination.index] = True
            for edge in node.incoming_edges:
                adjacency_matrix[node.index, edge.source.index] = True

        return graph_construction_matrix, adjacency_matrix

    def identify_token_ids(self, tokenized_input: BatchEncoding):
        remaining_text = self.text
        index = 0
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.copy_count > 0:
                    continue
                token.token_ids = []

                text_start_position = remaining_text.find(token.text)
                if text_start_position < 0:
                    print("token text couldn't be found")
                    print(f"token text: {token.text}")
                    print(f"remaining text: {remaining_text[:30]}...")
                    print(f"encompassing text: {self.text[(index - 15):(index + 15)]}")

                if text_start_position > 1 and not re.match("^\\s*" + re.escape(token.text),
                                                            remaining_text[:(text_start_position + 30)]):
                    print(f"token text has been found at position {text_start_position}")
                    print(f"token text: {token.text}")
                    print(f"remaining text: {remaining_text[:(text_start_position + 30)]}...")
                    print(f"encompassing text: {self.text[(index - 15):(index + 15)]}")

                remaining_text = remaining_text[(text_start_position + len(token.text)):]
                index += text_start_position
                text_positions = range(index, index + len(token.text))

                index += len(token.text)
                for position in text_positions:
                    token_id = tokenized_input.char_to_token(position)
                    if token_id is not None:
                        token.token_ids.append(token_id)

        last_original_token = None
        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.copy_count == 0:
                    last_original_token = token
                else:
                    token.token_ids = last_original_token.token_ids[:]
                    # token.original = last_original_token
                    # assert(token.text == last_original_token.text)
                    # assert(token.index_in_sentence == last_original_token.index_in_sentence)
                    # assert(token.index_in_document == last_original_token.index_in_document)
                    # assert(token.sentence_index == last_original_token.sentence_index)

    def print_all_entities_in_sentence(self, index):
        print(f"listing all entities for sentence {index}: {str(self.sentences[index])}")
        for cluster in self.entity_clusters:
            for mention in cluster.mentions:
                if mention.root.sentence_index == index:
                    print(f"{str(mention)} {str([token.index_in_sentence for token in mention.tokens])}")


    def create_two_hop_connections(self):
        two_hop_neighbours: set[(usg.Node, usg.Node)] = set()
        for node in self.unified_semantic_graph.nodes:
            neighbours = [e.destination for e in node.outgoing_edges]
            neighbours.extend([e.source for e in node.incoming_edges])

            for n1 in neighbours:
                for n2 in neighbours:
                    two_hop_neighbours.add((n1, n2) if n1.index <= n2.index else (n2, n1))

        for n1, n2 in two_hop_neighbours:
            self.unified_semantic_graph.add_edge(n1, n2, "TWO_HOP_NEIGHBOUR")


    def is_punctuation(self, node):
        return node.root.xpos.startswith((".", ",", "?", "!", ":")) or node.root.dependency_relation == "punct"


    def merge(self, doc: Document):
        self.text = self.text + " " + doc.text
        self.sentences.extend(doc.sentences)
        self.entity_clusters.extend(doc.entity_clusters)
        if doc.unified_semantic_graph:
            self.unified_semantic_graph = usg.Graph()
            self.unified_semantic_graph.nodes.extend(doc.unified_semantic_graph.nodes)
            self.unified_semantic_graph.edges.extend(doc.unified_semantic_graph.edges)
            self.unified_semantic_graph.roots.extend(doc.unified_semantic_graph.roots)
            self.unified_semantic_graph.removed_nodes.extend(doc.unified_semantic_graph.removed_nodes)
            self.unified_semantic_graph.removed_edges.extend(doc.unified_semantic_graph.removed_edges)
        self.tokens.extend(doc.tokens)

    @staticmethod
    def build_from_corenlpdatapoint(datapoint: datasets.CoreNlpDatapoint, with_graph_augmentations=True):
        document = Document(datapoint.document_text)
        document.unified_semantic_graph = usg.Graph()
        for doc in datapoint.annotation:
            if doc.unified_semantic_graph:
                document.unified_semantic_graph.nodes.extend(doc.unified_semantic_graph.nodes)
                document.unified_semantic_graph.edges.extend(doc.unified_semantic_graph.edges)
                document.unified_semantic_graph.roots.extend(doc.unified_semantic_graph.roots)

            for node in doc.unified_semantic_graph.nodes:
                for t in node.tokens:
                    t.node = node

                for tlist in node.replaced_tokens:
                    for t in tlist:
                        t.node = node

            for s in doc.sentences:
                document.sentences.append(s)
                document.tokens.extend(s.tokens)

        for index, token in enumerate(document.tokens):
            token.index_in_document = index

        for index, node in enumerate(document.unified_semantic_graph.nodes):
            node.index = index
            for t in node.tokens:
                t.node = node

        if with_graph_augmentations:
            document.create_two_hop_connections()
            document.create_supernode()

            document.unified_semantic_graph.nodes = document.unified_semantic_graph.nodes[
                                                    -1:] + document.unified_semantic_graph.nodes[0:-1]
            for index, node in enumerate(document.unified_semantic_graph.nodes):
                node.index = index

        return document


class Sentence:
    def __init__(self, doc: Document):
        self.document: Document = doc
        self.tokens: list[Token] = []
        self.semantic_graph: usg.Graph = None

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])

    @staticmethod
    def parse_from(s: Sentence, references):
        if s is None:
            return None
        elif s in references:
            return references[s]
        else:
            sentence = Sentence(references[s.document])
            references[s] = sentence
            sentence.tokens = [Token.parse_from(t, references) for t in s.tokens] if s.tokens else []
            sentence.semantic_graph = usg.Graph.parse_from(s.semantic_graph, references)
            return sentence


class Token:
    def __init__(self):
        self.text: str = None
        self.xpos: str = None
        self.copy_count: int = None
        self.begin_pos: int = None
        self.end_pos: int = None
        self.original: Token = None
        self.node: usg.Node = None
        self.dependency_head: Token = None
        self.dependency_children: list[Token] = []
        self.dependency_relation: str = None
        self.index_in_sentence: int = None
        self.index_in_document: int = None
        self.sentence_index: int = None
        self.token_ids: list[int] = []
        self.sentence: Sentence = None

    def __str__(self):
        return self.text

    @staticmethod
    def parse_from(t: Token, references):
        if t is None:
            return None
        elif t in references:
            return references[t]
        else:
            token = Token()
            references[t] = token
            token.text = str(t.text)
            token.xpos = str(t.xpos)
            token.copy_count = int(t.copy_count)
            token.begin_pos = int(t.begin_pos)
            token.end_pos = int(t.end_pos)
            token.original = Token.parse_from(t.original, references)
            token.node = usg.Node.parse_from(t.node, references)
            token.dependency_head = Token.parse_from(t.dependency_head, references)
            token.dependency_children = [Token.parse_from(c, references) for c in t.dependency_children]
            token.dependency_relation = str(t.dependency_relation)
            token.index_in_sentence = int(t.index_in_sentence)
            token.index_in_document = int(t.index_in_document)
            token.sentence_index = int(t.sentence_index)
            token.sentence = Sentence.parse_from(t.sentence, references)
            return token


class Entity:
    def __init__(self):
        self.tokens: list[Token] = []
        self.root: Token = None
        self.superset_entity: Entity = None

    @staticmethod
    def parse_from(e: Entity, references):
        if e is None:
            return None
        elif e in references:
            return references[e]
        else:
            entity = Entity()
            references[e] = entity
            entity.tokens = [Token.parse_from(t, references) for t in e.tokens]
            entity.root = Token.parse_from(e.root, references)
            return entity

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])


class EntityCluster:
    def __init__(self):
        self.representative: Entity = None  # the representative among the mentions
        self.mentions: list[Entity] = []

    @staticmethod
    def parse_from(c: EntityCluster, references):
        if c is None:
            return None
        elif c in references:
            return references[c]
        else:
            cluster = EntityCluster()
            references[c] = cluster
            cluster.representative = Entity.parse_from(c.representative, references)
            cluster.mentions = [Entity.parse_from(e, references) for e in c.mentions]
            return cluster

    def __str__(self):
        return str(self.representative)
