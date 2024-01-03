/**
 * Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
 * Author: Osman Alperen Koras
 * Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
 * Email: osman.koras@uni-due.de
 * Date: 12.10.2023
 *
 * License: MIT
 */
package semanticgraph;

import bass.model.KnowledgeGraphCoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;


public class Document implements Serializable {
    public List<Sentence> sentences = new ArrayList<Sentence>();
    public Graph unified_semantic_graph = null;
    public List<EntityCluster> entity_clusters = new ArrayList<EntityCluster>();
    public String text = null;

    public Document(bass.data.Document document, String text) {
        this.text = text;
        for (var sentence : document.getSentences()) {
            this.sentences.add(parse_sentence(sentence));
        }
        unified_semantic_graph = parse_graph(document);

        int di = 0;
        for (Sentence sentence : sentences) {
            for (int ti = 0; ti < sentence.tokens.size(); ti++) {
                sentence.tokens.get(ti).index_in_document = di;
                sentence.tokens.get(ti).index_in_sentence = sentence.tokens.get(ti).index_in_sentence-1;
                assert(sentence.tokens.get(ti).index_in_sentence >= 0);
                di++;
            }
        }

    }

    private Sentence parse_sentence(bass.data.Sentence s) {
        var sentence = new Sentence();
        var mapping = new HashMap<IndexedWord, Token>();
        for (var t : s.getTokens()) {
            var token = parse_token(t);
            mapping.put(t, token);
            token.sentence = sentence;
            sentence.tokens.add(token);
        }

        for(var token : sentence.tokens) {
            if(token.original != null) {
                token.original =  mapping.get((IndexedWord) token.original);
            }
        }

        sentence.document = this;
        return sentence;
    }

    private Token parse_token(IndexedWord t) {
        var token = new Token();
        token.text = t.originalText();
        token.xpos = t.tag();
        token.index_in_sentence = t.index();
        token.sentence_index = t.sentIndex();
        token.copy_count = t.copyCount();
        token.original = t.getOriginal();
        if (t.containsKey(CoreAnnotations.CodepointOffsetBeginAnnotation.class) &&
            t.containsKey(CoreAnnotations.CodepointOffsetEndAnnotation.class)) {
            token.begin_pos = t.get(CoreAnnotations.CodepointOffsetBeginAnnotation.class);
            token.end_pos = t.get(CoreAnnotations.CodepointOffsetEndAnnotation.class);
        }
        return token;
    }

    private Graph parse_graph(bass.data.Document document) {
        var graph = new Graph();

        Map<String, Node> nodeMapping = new HashMap<>();

        var i = 0;
        for (var key : document.SentenceProj.keySet()) {
            var node = parse_node(document.SentenceProj.get(key));
            node.index = i;
            i++;
            graph.nodes.add(node);
            nodeMapping.put(key, node);
        }

        for (var edge : document.getSemanticGraph().edgeIterable()) {
            graph.edges.add(parse_edge(nodeMapping, edge));
        }

        for (var r : document.RootNodes) {
            var root = nodeMapping.get(r);
            graph.roots.add(root);
            var root_edge = new Edge();
            root_edge.relation = "ROOT";
            root_edge.source = root;
            root_edge.destination = root;
            root_edge.is_extra = false;
            graph.edges.add(root_edge);
        }

        return graph;
    }


    private Node parse_node(List<List<Integer>> lists) {
        Function<List<Integer>, List<Token>> tokenIndicesToSentenceTokens = (List<Integer> list) -> new ArrayList<Token>(this.sentences.get(list.get(1)).tokens.stream().filter(t -> list.subList(2, list.size()).contains(t.index_in_sentence)).toList());

        var node = new Node();
        node.tokens.addAll(tokenIndicesToSentenceTokens.apply(lists.get(0)));
        node.root = node.tokens.stream().filter(t -> lists.get(0).subList(2, lists.get(0).size()).contains(t.index_in_sentence)).findFirst().get();
        for (int i = 1; i < lists.size(); i++) {
            node.replaced_tokens.add(tokenIndicesToSentenceTokens.apply(lists.get(i)));
        }
        assert node.root.node == null;
        node.root.node = node;
        return node;
    }

    private Edge parse_edge(Map<String, Node> nodeMapping, SemanticGraphEdge e) {
        var edge = new Edge();
        edge.relation = e.getRelation().toString();
        edge.is_extra = e.isExtra();
        edge.weight = e.getWeight();
        edge.destination = nodeMapping.get(e.getTarget().containsKey(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class)
                ? e.getTarget().get(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class)
                : e.getTarget().word());
        edge.source = nodeMapping.get(e.getSource().containsKey(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class)
                ? e.getSource().get(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class)
                : e.getSource().word());
        edge.source.outgoing_edges.add(edge);
        edge.destination.incoming_edges.add(edge);
        return edge;
    }

    public Document(CoreDocument document) {
        // Do not run this code, parse_token(CoreLabel t) still needs to be fixed! the index_in_sentence must be set the
        // same as in CoreNLP (starting at 1) and only later re-indexed to start at 0
        assert(false);
        text = document.text();
        for (var sentence : document.sentences()) {
            this.sentences.add(parse_sentence(sentence));
        }
        int di = 0;
        for (Sentence sentence : sentences) {
            for (int ti = 0; ti < sentence.tokens.size(); ti++) {
                sentence.tokens.get(ti).index_in_document = di;
                di++;
            }
        }

        for (var corefChain : document.corefChains().values()) {
            var cluster = new EntityCluster();
            for (var mention : corefChain.getMentionsInTextualOrder()) {
                var entity = new Entity();
                var sentence = this.sentences.get(mention.sentNum - 1);
                entity.tokens = new ArrayList<>(sentence.tokens.subList(mention.startIndex - 1, mention.endIndex - 1));
                entity.root = sentence.tokens.get(mention.headIndex);
                cluster.mentions.add(entity);
                if (mention == corefChain.getRepresentativeMention()) {
                    cluster.representative = entity;
                }
            }
            this.entity_clusters.add(cluster);
        }
    }

    private Sentence parse_sentence(CoreSentence s) {
        var sentence = new Sentence();
        for (var t : s.tokens()) {
            sentence.tokens.add(parse_token(t));
        }

        for (int ti = 0; ti < sentence.tokens.size(); ti++) {
            sentence.tokens.get(ti).sentence_index = this.sentences.size();
            sentence.tokens.get(ti).index_in_sentence = ti;
        }

        sentence.semantic_graph = parse_graph(sentence, s.dependencyParse());
        sentence.document = this;

        for (var token : sentence.tokens) {
            if (!token.node.incoming_edges.isEmpty()) {
                token.dependency_head = token.node.incoming_edges.get(0).source.root;
                token.dependency_relation = token.node.incoming_edges.get(0).relation;
            }
            for (var edge : token.node.outgoing_edges) {
                token.dependency_children.add(edge.destination.root);
            }
            token.sentence = sentence;
        }

        return sentence;
    }

    private Token parse_token(CoreLabel t) {
        var token = new Token();
        token.text = t.originalText();
        token.xpos = t.tag();
        token.index_in_sentence = t.index();
        token.sentence_index = t.sentIndex();
        // token.copy_count = t.copyCount();
        // token.original = t.getOriginal();
        if (t.containsKey(CoreAnnotations.CodepointOffsetBeginAnnotation.class) &&
                t.containsKey(CoreAnnotations.CodepointOffsetEndAnnotation.class)) {
            token.begin_pos = t.get(CoreAnnotations.CodepointOffsetBeginAnnotation.class);
            token.end_pos = t.get(CoreAnnotations.CodepointOffsetEndAnnotation.class);
        }
        return token;
    }

    private Graph parse_graph(Sentence sentence, SemanticGraph dependencyParse) {
        var graph = new Graph();
        for (var word : dependencyParse.vertexSet()) {
            graph.nodes.add(parse_node(sentence, word));
        }

        graph.nodes.sort(Comparator.comparingInt(n -> n.index));
        for (var edge : dependencyParse.edgeIterable()) {
            graph.edges.add(parse_edge(graph, edge));
        }

        for (var r : dependencyParse.getRoots()) {
            var root = graph.nodes.get(r.index() - 1);
            graph.roots.add(root);
            var root_edge = new Edge();
            root_edge.relation = "ROOT";
            root_edge.source = root;
            root_edge.destination = root;
            root_edge.is_extra = false;
            graph.edges.add(root_edge);
        }

        return graph;
    }

    private Node parse_node(Sentence sentence, IndexedWord word) {
        var node = new Node();
        node.index = word.index() - 1;
        node.root = sentence.tokens.get(node.index);
        node.tokens.add(node.root);
        assert node.root.node == null;
        node.root.node = node;
        return node;
    }

    private Edge parse_edge(Graph graph, SemanticGraphEdge e) {
        var edge = new Edge();
        edge.relation = e.getRelation().toString();
        edge.is_extra = e.isExtra();
        edge.destination = graph.nodes.get(e.getTarget().index() - 1);
        edge.source = graph.nodes.get(e.getSource().index() - 1);
        edge.source.outgoing_edges.add(edge);
        edge.destination.incoming_edges.add(edge);
        return edge;
    }
}

