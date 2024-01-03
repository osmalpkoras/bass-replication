# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

from pyvis import network as net

from preprocessing import doc


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.removed_nodes: list[Node] = []
        self.removed_edges: list[Edge] = []
        self.roots: list[Node] = []

    @staticmethod
    def parse_from(g: Graph, references):
        if g is None:
            return None
        elif g in references:
            return references[g]
        else:
            graph = Graph()
            references[g] = graph
            graph.nodes = [Node.parse_from(n, references) for n in g.nodes] if g.nodes else []
            graph.edges = [Edge.parse_from(e, references) for e in g.edges] if g.edges else []
            graph.roots = [Node.parse_from(r, references) for r in g.roots] if g.roots else []
            return graph

    def change_edge_direction(self, edge: Edge):
        if edge not in self.edges:
            return
        edge.source.outgoing_edges.remove(edge)
        edge.destination.incoming_edges.remove(edge)

        node = edge.destination
        edge.destination = edge.source
        edge.source = node

        edge.destination.incoming_edges.append(edge)
        edge.source.outgoing_edges.append(edge)

    def add_edge(self, source: Node, destination: Node, relation: str):
        edge = Edge()
        edge.relation = relation
        edge.source = source
        edge.destination = destination
        source.outgoing_edges.append(edge)
        destination.incoming_edges.append(edge)
        if relation == "ROOT" and edge.source not in self.roots:
            self.roots.append(source)
        self.edges.append(edge)

    def remove_edge(self, edge: Edge):
        edge.source.outgoing_edges.remove(edge)
        edge.destination.incoming_edges.remove(edge)
        self.edges.remove(edge)
        self.removed_edges.append(edge)
        if edge.relation == "ROOT" and edge.source in self.roots:
            self.roots.remove(edge.source)

    def swap_nodes(self, a: Node, b: Node):
        a_outgoing_edges = a.outgoing_edges[:]
        a_incoming_edges = a.incoming_edges[:]
        b_outgoing_edges = b.outgoing_edges[:]
        b_incoming_edges = b.incoming_edges[:]

        for edge in a_outgoing_edges:
            edge.source = b
        for edge in a_incoming_edges:
            edge.destination = b
        for edge in b_outgoing_edges:
            edge.source = a
        for edge in b_incoming_edges:
            edge.destination = a

        b.outgoing_edges = a_outgoing_edges
        b.incoming_edges = a_incoming_edges
        a.incoming_edges = b_incoming_edges
        a.outgoing_edges = b_outgoing_edges

    def remove_node(self, node: Node):
        self.nodes.remove(node)
        if node in self.roots:
            self.roots.remove(node)
        self.removed_nodes.append(node)

        for token in node.tokens:
            token.node = None
        for replaced_tokens in node.replaced_tokens:
            for token in replaced_tokens:
                token.node = None

        for edge in node.incoming_edges[:]:
            self.remove_edge(edge)

        for edge in node.outgoing_edges[:]:
            self.remove_edge(edge)

    def change_edge_source(self, edge: Edge, new_source: Node):
        if edge not in self.edges or new_source not in self.nodes:
            return

        if edge.relation == "ROOT":
            self.remove_edge(edge)
            self.add_edge(new_source, new_source, "ROOT")
        else:
            edge.source.outgoing_edges.remove(edge)
            edge.source = new_source
            new_source.outgoing_edges.append(edge)

    def change_edge_destination(self, edge: Edge, new_destination: Node):
        if edge not in self.edges or new_destination not in self.nodes:
            return
        if edge.relation == "ROOT":
            self.remove_edge(edge)
            self.add_edge(new_destination, new_destination, "ROOT")
        else:
            edge.destination.incoming_edges.remove(edge)
            edge.destination = new_destination
            new_destination.incoming_edges.append(edge)

    def merge_nodes(self, remaining_node: Node, consumed_node: Node, replace_tokens=False, transfer_edges=True):
        if remaining_node == consumed_node:
            return
        if remaining_node not in self.nodes or consumed_node not in self.nodes:
            return

        if replace_tokens:
            remaining_node.replaced_tokens.append(consumed_node.tokens[:])
        else:
            # add all tokens of the consumed node to the remaining node, then sort by indices
            remaining_node.tokens.extend(consumed_node.tokens)
            remaining_node.tokens.sort(key=lambda x: x.index_in_sentence)

        remaining_node.replaced_tokens.extend(consumed_node.replaced_tokens)
        # for all tokens of the consumed node, set the remaining node as their new node
        for token in consumed_node.tokens:
            token.node = remaining_node
        for replaced_tokens in consumed_node.replaced_tokens:
            for token in replaced_tokens:
                token.node = remaining_node

        if transfer_edges:
            node_neighbours = [edge.destination for edge in remaining_node.outgoing_edges]
            node_neighbours.extend([edge.source for edge in remaining_node.incoming_edges])
            node_neighbours.append(remaining_node)
            # redirect all incoming edges from the consumed node to the remaining node,
            # but delete any edges inbetween these two nodes
            for edge in consumed_node.incoming_edges[:]:
                if edge.source in node_neighbours:
                    self.remove_edge(edge)
                else:
                    self.change_edge_destination(edge, remaining_node)

            # redirect all outgoing edges from the consumed node to the remaining node,
            # but delete any edges inbetween these two nodes
            for edge in consumed_node.outgoing_edges[:]:
                if edge.destination in node_neighbours:
                    self.remove_edge(edge)
                else:
                    self.change_edge_source(edge, remaining_node)
        else:
            for edge in consumed_node.incoming_edges[:]:
                self.remove_edge(edge)
            for edge in consumed_node.outgoing_edges[:]:
                self.remove_edge(edge)

        self.nodes.remove(consumed_node)
        self.removed_nodes.append(consumed_node)
        if consumed_node in self.roots:
            self.roots.remove(consumed_node)
            self.roots.append(remaining_node)

    def show(self, path, r=False):
        nodes = self.nodes[:]
        edges = self.edges[:]
        if r:
            edges = []
            for node in nodes:
                edges.extend(node.outgoing_edges)
                edges.extend(node.incoming_edges)
            edges = set(edges)

        g = net.Network(height='100%', width='100%', heading='', directed=True)
        for node in nodes:
            g.add_node(node.index, str(node), title=node.root.xpos)
        for edge in edges:
            g.add_edge(edge.source.index, edge.destination.index, label=edge.relation)
        g.show(path)

    def collapse_node(self, node):
        assert len(node.incoming_edges) == 1
        assert node.incoming_edges[0].relation != "ROOT"

        for edge in node.outgoing_edges:
            self.change_edge_source(edge, node.incoming_edges[0].source)

        self.remove_node(node)


class Node:
    def __init__(self):
        self.index: int = None
        self.root: doc.Token = None
        self.tokens: list[doc.Token] = []
        self.replaced_tokens: list[list[doc.Token]] = []
        self.incoming_edges: list[Edge] = []
        self.outgoing_edges: list[Edge] = []

    @staticmethod
    def parse_from(n: Node, references):
        if n is None:
            return None
        elif n in references:
            return references[n]
        else:
            node = Node()
            references[n] = node
            node.index = int(n.index)
            node.root = doc.Token.parse_from(n.root, references)
            node.tokens = [doc.Token.parse_from(c, references) for c in n.tokens]
            node.incoming_edges = [Edge.parse_from(e, references) for e in n.incoming_edges]
            node.outgoing_edges = [Edge.parse_from(e, references) for e in n.outgoing_edges]
            return node

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])


class Edge:

    def __init__(self):
        self.source: Node = None
        self.destination: Node = None
        self.relation: str = None
        self.is_extra: str = None
        self.weight: float = None

    @staticmethod
    def parse_from(e: Edge, references):
        if e is None:
            return None
        elif e in references:
            return references[e]
        else:
            edge = Edge()
            references[e] = edge
            edge.source = Node.parse_from(e.source, references)
            edge.destination = Node.parse_from(e.destination, references)
            edge.relation = str(e.relation)
            edge.is_extra = str(e.is_extra)
            edge.weight = float(e.weight)
            return edge


    def __str__(self):
        return f"({self.source}) {self.relation} -> ({self.destination})"


