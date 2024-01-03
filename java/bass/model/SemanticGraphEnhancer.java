/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import bass.data.Coreference;
import bass.data.Document;
import bass.data.Mention;
import bass.data.Sentence;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.StringUtils;



/*
 * Performs several enhancements of semantic graphs to make them a better semantic representation.
 * Assumes that the graph contains UniversalEnglishGrammaticalRelations.
 *
 * @author Wei Li
 */

public class SemanticGraphEnhancer {

    /* Resolves pronouns in the semantic graph and replaces the pronoun node in
     * the semantic graph with the node of its antecedent.
     *
     * @param sg A SemanticGraph.
     */
    public static void Corefresolve(Document doc) {
        List<Coreference> corefs = doc.getCoreferences();
        List<Sentence> sentences = doc.getSentences();
        String SpecialToken = "#/#";
        //collapse concept in coreference chiains
        for (int i = 0; i < corefs.size(); i++) {
            Coreference cof = corefs.get(i);
            ArrayList<Mention> mentions = cof.getMentions();
            List<IndexedWord> target_compoud = new ArrayList<IndexedWord>();
            IndexedWord head_concept = new IndexedWord();
            String target_word = "";
            for (int j = 0; j < mentions.size(); j++) {
                Mention mentionj = mentions.get(j);
                int sen_idx = mentionj.getSentence_idx();
                int start_idx = mentionj.getWord_start_idx();
                int end_idx = mentionj.getWord_end_idx();
                int head_idx = mentionj.getHead_idx();
                SemanticGraph repSg = sentences.get(sen_idx).getSemanticGraph();
                IndexedWord head_token = repSg.getNodeByIndexSafe(head_idx);
                if (head_token == null) continue;
                //System.out.println(head_token);
                List<IndexedWord> compound = Generics.newArrayList();

                List<SemanticGraphEdge> children_edges = repSg.outgoingEdgeList(head_token);
                compound.add(head_token);
                if (end_idx - head_idx > 5) {
                    for (SemanticGraphEdge edge : children_edges) {
                        IndexedWord child = edge.getDependent();
                        if (child.index() >= end_idx || child.index() < start_idx) continue;
                        String rel_str = edge.getRelation().toString();
                        if (rel_str.startsWith("compound") || rel_str.startsWith("name") || rel_str.startsWith("amod")
                                || rel_str.equals("det") || rel_str.startsWith("nummod") || rel_str.startsWith("predet")
                                || rel_str.startsWith("advmod") || rel_str.startsWith("dep")) {
                            compound.add(child);
                        }
                    }
                } else
                    for (int idx = start_idx; idx < end_idx; idx++) {
                        IndexedWord child_index = repSg.getNodeByIndexSafe(idx);
                        if (child_index != null && child_index != head_token) compound.add(child_index);
                    }
                //save projection and change all the coref nodes to the first one
                //System.out.println(compound);
                if (compound.size() == 0) continue;
                generate_new_node(head_token, compound, repSg, false);


                if (target_compoud.size() == 0) {
                    target_word = head_token.get(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class);
                    //System.out.println(compound);
                    for (IndexedWord c : compound) target_compoud.add(c.makeCopy());
                    head_concept = head_token;
                }
                //change other nodes to the same one

                if (target_compoud.size() == 0) continue;
                //System.out.println(compound);
                head_token.set(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class,
                        StringUtils.join(target_compoud.stream().map(IndexedWord::word), " "));
                head_token.set(KnowledgeGraphCoreAnnotations.CompoundLemmaAnnotation.class, SpecialToken);
                head_token.set(KnowledgeGraphCoreAnnotations.CompoundIndexedWordAnnotation.class, target_compoud);

                List<Integer> ConceptTokens = new ArrayList<Integer>();
                ConceptTokens.add(doc.docNo);
                ConceptTokens.add(sen_idx);


                for (IndexedWord c : compound) ConceptTokens.add(c.index());

                if (!doc.SentenceProj.containsKey(target_word))
                    doc.SentenceProj.put(target_word, new ArrayList<List<Integer>>());
                doc.SentenceProj.get(target_word).add(ConceptTokens);

                List<SemanticGraphEdge> incomingEdges = repSg.getIncomingEdgesSorted(head_token);
                for (SemanticGraphEdge edge : incomingEdges) {
                    repSg.removeEdge(edge);
                    repSg.addEdge(edge.getGovernor(), head_concept, edge.getRelation(), edge.getWeight(),
                            edge.isExtra());
                }


                if (head_token != head_concept) repSg.removeVertex(head_token);
            }

        }

    }

    public static boolean isCopulaVerb(IndexedWord vertex, SemanticGraph sg) {
        //System.out.println(vertex);
        //System.out.println(sg.relns(vertex));
        try {
            Set<GrammaticalRelation> relns = sg.relns(vertex);
            if (relns.isEmpty())
                return false;
            //e.g. Bill is big
            boolean result = relns.contains(UniversalEnglishGrammaticalRelations.COPULA);
            return result;
        } catch (Exception e) {
            return false;
        }

    }

    /* Performs all the enhancements in the following order:
     * <ol>
     * <li>For verbs, collapse verbs with particles, neg, aux and mwe </li>
     * <li>For nouns, collapse nouns with compound, det, det:predet, amod, nummod, neg, dep, </li>
     * </ol>
     * @param sg
     */
    public static void enhance(SemanticGraph sg) {
        if (sg == null)
            return;

        Collection<IndexedWord> roots = sg.getRoots();
        for (IndexedWord root : roots)
            try {
                List<IndexedWord> visited = Generics.newArrayList();
                ;
                enhance(root, sg, visited);
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
    }

    public static void generate_new_node(IndexedWord word, List<IndexedWord> compound, SemanticGraph sg, Boolean setChildrenEdges) {
        Collections.sort(compound);

        //combined specific children to the node of word
        String lemma = StringUtils
                .join(compound.stream().map(x -> x.lemma() != null ? x.lemma() : x.word()), " ");
        word.set(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class,
                StringUtils.join(compound.stream().map(IndexedWord::word), " "));
        word.set(KnowledgeGraphCoreAnnotations.CompoundLemmaAnnotation.class, lemma);
        word.set(KnowledgeGraphCoreAnnotations.CompoundIndexedWordAnnotation.class, compound);

        //remove collapsed children of words from sg, all of them has been combined into the node of word
        for (IndexedWord word2 : compound) {

            if (!word2.equals(word)) {
                if (setChildrenEdges) {
                    List<SemanticGraphEdge> outgoingEdges = sg.getOutEdgesSorted(word2);
                    for (SemanticGraphEdge edge : outgoingEdges) {
                        sg.removeEdge(edge);
                        sg.addEdge(word, edge.getTarget(), edge.getRelation(), edge.getWeight(), edge.isExtra());
                    }
                }
                sg.removeVertex(word2);
            }
        }
    }

    public static void remove_cases_child(IndexedWord word, SemanticGraph sg) {
        if (!sg.hasChildren(word) || word == null) return;
        List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(word);
        for (SemanticGraphEdge edge : children_edges) {
            String rel_str = edge.getRelation().toString();
            if (rel_str.startsWith("case") || rel_str.startsWith("aux")
                    || rel_str.equals("cc") || rel_str.startsWith("mark")) {

                sg.removeVertex(edge.getDependent());
                sg.removeEdge(edge);
            }

        }
        //generate_new_node(word,compound,sg);

    }

    public static void enhance(IndexedWord root, SemanticGraph sg, List<IndexedWord> visited) throws Exception {
        if (root == null || sg == null || visited.contains(root))
            return;
        visited.add(root);

        //System.out.println(root);
        if (root.tag() == null) {
            return;
        }
        if ((root.tag() != null) && root.get(KnowledgeGraphCoreAnnotations.CompoundLemmaAnnotation.class) == null) {
            if (root.tag().startsWith("VB")) {
                if (isCopulaVerb(root, sg)) {
                    IndexedWord parent = sg.getParent(root);
                    collapse_Is(root, parent, sg, visited);
                } else
                    collapseVerb(root, sg);
            } else if (root.tag().startsWith("NN") || root.tag().startsWith("JJ"))
                collapseNounPhrase(root, sg, visited);
            else if (root.tag().startsWith("CD"))
                collapseCDPhrase(root, sg, visited);
            else if (root.tag().startsWith(".") || root.tag().startsWith(",") || root.tag().startsWith("?")
                    || root.tag().startsWith("!") || root.tag().startsWith(":")) {
                List<SemanticGraphEdge> punct_parent_edges = sg.incomingEdgeList(root);
                for (SemanticGraphEdge pedge : punct_parent_edges) {
                    sg.removeEdge(pedge);
                }
                sg.removeVertex(root);
                return;
            } else {
                List<IndexedWord> compound = Generics.newArrayList();
                compound.add(root);
                List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(root);
                for (SemanticGraphEdge edge : children_edges) {
                    String rel_str = edge.getRelation().toString();
                    if (rel_str.startsWith("mwe")) {
                        compound.add(edge.getDependent());
                        sg.removeEdge(edge);
                    }
                    generate_new_node(root, compound, sg, false);
                }
            }
        }


        if (sg.hasChildren(root)) {
            for (IndexedWord child : sg.getChildList(root))
                enhance(child, sg, visited);
            remove_cases_child(root, sg);
        }
    }

    /* Collapses verbs with verbal particles and auxiliaries
     * @param sg */

    public static void collapseVerb(IndexedWord word, SemanticGraph sg) {
        List<IndexedWord> compound = Generics.newArrayList();
        compound.add(word);
        List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(word);
        for (SemanticGraphEdge edge : children_edges) {
            IndexedWord child = edge.getDependent();
            String rel_str = edge.getRelation().toString();
            if (rel_str.equals("compound:prt") || rel_str.equals("aux") || rel_str.startsWith("auxpass")
                    || rel_str.startsWith("neg") || rel_str.startsWith("expl") || rel_str.startsWith("advmod")) {
                compound.add(child);
                sg.removeEdge(edge); // remove the edge from semantic graph
            }
        }

        generate_new_node(word, compound, sg, true);
    }

    /* First, collapses COPULA verbs with auxiliaries and negation
     * Then, move COPULA verb to its parent position, and move its parent to be its child
     * E.g. The great prize [was] for his explanation of the photoelectric effect.
     */

    public static List<IndexedWord> collapse_Is(IndexedWord word, IndexedWord parent, SemanticGraph sg, List<IndexedWord> visited) {

        List<IndexedWord> compound = Generics.newArrayList();
        compound.add(word);
        List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(parent);

        for (SemanticGraphEdge edge : children_edges) {
            String rel_str = edge.getRelation().toString();
            IndexedWord child = edge.getDependent();
            if (rel_str.startsWith("nsubj")) {
                sg.addEdge(word, edge.getTarget(), edge.getRelation(), edge.getWeight(), edge.isExtra());
                sg.removeEdge(edge);
            }
            if (rel_str.startsWith("mwe") || rel_str.startsWith("aux") || rel_str.startsWith("case")
                    || rel_str.startsWith("mark") || rel_str.startsWith("neg")) {
                if (child.index() > word.index()) {
                    compound.add(child);
                    sg.removeEdge(edge);
                }
            }
        }

        List<SemanticGraphEdge> parent_edges = sg.incomingEdgeList(parent);
        for (SemanticGraphEdge edge : parent_edges) {
            String rel_str = edge.getRelation().toString();
            if (rel_str.startsWith("ccomp") || rel_str.startsWith("xcomp")) {
                sg.addEdge(edge.getGovernor(), word, edge.getRelation(), edge.getWeight(), edge.isExtra());
                sg.removeEdge(edge);
            }

        }
        generate_new_node(word, compound, sg, true);
        return compound;

    }

    /* Collapses noun phrase with compound, det, name, amod etc.
     * @param sg */
    public static void collapseNounPhrase(IndexedWord word, SemanticGraph sg, List<IndexedWord> visited) {

        List<IndexedWord> compound = Generics.newArrayList();
        compound.add(word);

        List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(word);
        //System.out.println(children_edges);
        for (SemanticGraphEdge edge : children_edges) {
            IndexedWord child = edge.getDependent();
            String rel_str = edge.getRelation().toString();
            if (rel_str.startsWith("compound") || rel_str.startsWith("name")
                    || rel_str.startsWith("amod") || rel_str.equals("det")
                    || rel_str.startsWith("nummod") || rel_str.startsWith("predet")
                    || rel_str.startsWith("advmod")
            ) {


                //compound.addAll(sg.getSubgraphVertices(child)); //add all decendant nodes of child
                //!!! do not merge the
                compound.add(child);
                sg.removeEdge(edge);
                //System.out.println(sg.getSubgraphVertices(child));

                // remove the edge from semantic graph
            }
        }
        if (word.get(KnowledgeGraphCoreAnnotations.CompoundLemmaAnnotation.class) == null)
            generate_new_node(word, compound, sg, true);

    }

    /* Collapses CD phrase with mwe, det:qmod etc.
     * @param sg */

    public static IndexedWord collapseCDPhrase(IndexedWord word, SemanticGraph sg, List<IndexedWord> visited) {
        List<IndexedWord> compound = Generics.newArrayList();
        compound.add(word);

        // combine mwe children, e,g, one of
        List<SemanticGraphEdge> children_edges = sg.outgoingEdgeList(word);
        for (SemanticGraphEdge edge : children_edges) {
            IndexedWord child = edge.getDependent();
            String rel_str = edge.getRelation().toString();
            if (rel_str.startsWith("mwe")) {
                compound.add(child);
                sg.removeEdge(edge);
            }
        }

        // combine det:qmod parent, e,g, one of the two pillars
        generate_new_node(word, compound, sg, true);
        return word;

    }

    /*
     *
     * Performs all the enhancements in the following order:
     * <ol>
     * <li>Collapse verbs with particles, and compound nouns.</li>
     * <li>Resolve pronouns.</li>
     * </ol>
     *
     * @param sg
     */
    public static void enhance(Document doc) {
        List<Sentence> sentences = doc.getSentences();
        // Collapse verbs with particles, and compound nouns.
        //resolvePronouns(doc);
        //resolveCoref(doc);
        Corefresolve(doc);
        int docNo = doc.docNo;
        for (int i = 0; i < sentences.size(); i++) {
            SemanticGraph sg = sentences.get(i).getSemanticGraph();
            enhance(sg);
            Collection<IndexedWord> roots = sg.getRoots();
            for (IndexedWord root : roots) {
                if (root.containsKey(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class))
                    doc.RootNodes.add(root.get(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class));
                else
                    doc.RootNodes.add(root.word());
            }
            for (IndexedWord node : sg.vertexListSorted()) {
                List<Integer> ConceptTokens = new ArrayList<Integer>();
                ConceptTokens.add(docNo);
                ConceptTokens.add(i);

                if (node.containsKey(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class)) {
                    String target_word = node.get(KnowledgeGraphCoreAnnotations.CompoundWordAnnotation.class);
                    //((sen.no,w1,w2),(sen,w2,w3))

                    List<IndexedWord> compund = node.get(KnowledgeGraphCoreAnnotations.CompoundIndexedWordAnnotation.class);
                    String test_lemma = node.getString(KnowledgeGraphCoreAnnotations.CompoundLemmaAnnotation.class);
                    if (test_lemma.equals("#/#")) continue;
                    for (IndexedWord token : compund) {
                        ConceptTokens.add(token.index());
                    }
                    if (!doc.SentenceProj.containsKey(target_word))
                        doc.SentenceProj.put(target_word, new ArrayList<List<Integer>>());
                    doc.SentenceProj.get(target_word).add(ConceptTokens);

                } else {
                    String target_word = node.word();
                    if (!doc.SentenceProj.containsKey(target_word))
                        doc.SentenceProj.put(target_word, new ArrayList<List<Integer>>());
                    ConceptTokens.add(node.index());
                    doc.SentenceProj.get(target_word).add(ConceptTokens);
                }

            }

        }


    }
}
