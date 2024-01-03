/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.util.MapFactory;

/**
 * @author Wei Li
 */

public class KnowledgeGraph implements Serializable {

    private static final long serialVersionUID = -8192827831219296966L;
    private final DirectedMultiGraph<Concept, ConceptRelation> kg;
    private static final MapFactory<Concept, Map<Concept, List<ConceptRelation>>>
    		outerMapFactory = MapFactory.hashMapFactory();
    private static final MapFactory<Concept, List<ConceptRelation>>
    		innerMapFactory = MapFactory.hashMapFactory();

    public KnowledgeGraph() {
        kg = new DirectedMultiGraph<Concept, ConceptRelation>(
                outerMapFactory, innerMapFactory);
    }

    public int addEdge(Concept source, Concept target, List<IndexedWord> relation) {
        //relation = relation.replaceAll("^be ", "");
        ConceptRelation edge = new ConceptRelation(source, target, relation);

        List<ConceptRelation> rels = this.kg.getEdges(source, target);
        if (rels != null && !rels.isEmpty())
        {
        	for(ConceptRelation rel:rels)
            {
            	if(rel.equals(edge))
            		return 0;
            }
        }

        this.kg.add(source, target, edge);
        return 1;
    }

    public int addEdge(Concept source, Concept target, ConceptRelation relation) {
        List<ConceptRelation> rels = this.kg.getEdges(source, target);
        if (rels != null && !rels.isEmpty())
        {
        	for(ConceptRelation rel:rels)
            {
            	if(rel.equals(relation))
            		return 0;
            }
        }

        this.kg.add(source, target, relation);
        return 1;
    }

    public List<ConceptRelation> relationListSorted() {
        List<ConceptRelation> relations = new ArrayList<ConceptRelation>(
                this.kg.getAllEdges());
        Collections.sort(relations);
        return relations;
    }

    public List<Concept> nodeListSorted() {
        ArrayList<Concept> nodes = new ArrayList<Concept>(
                this.kg.getAllVertices());
        Collections.sort(nodes);
        return nodes;
    }

    public String toReadableString() {
        StringBuilder buf = new StringBuilder();
        buf.append(String.format("%-20s%-20s%-20s%n", "source", "reln", "target"));
        buf.append(String.format("%-20s%-20s%-20s%n", "---", "----", "---"));
        for (ConceptRelation edge : this.relationListSorted()) {
            buf.append(String.format("%-20s%-20s%-20s%n", edge.getSource(),
                    edge.getRelationString(), edge.getTarget()));
        }

        buf.append(String.format("%n%n"));
        buf.append(String.format("%-20s%n", "Nodes"));
        buf.append(String.format("%-20s%n", "---"));

        for (Concept node : this.nodeListSorted()) {
            buf.append(String.format("%-20s%n", node));
			/*
			 * for (EventArgument arg : node.getArgs()) {
			 * buf.append(String.format("  -%-20s%n", arg)); }
			 */
        }

        return buf.toString();
    }

    public void addNode(Concept node) {
        this.kg.addVertex(node);
    }

    public Concept getOrAddNode(Concept node) {
        for (Concept node2 : this.kg.getAllVertices()) {
            if (node2.equals(node)) {
                return node2;
            }
        }
        this.kg.addVertex(node);
        return node;
    }

    //combine two knowledge graphs
    public void addChildGraph(KnowledgeGraph eg) {
        if (eg == null)
            return;

        List<ConceptRelation> relations = eg.relationListSorted();
        List<Concept> nodes = eg.nodeListSorted();

        for (int i = 0; i < relations.size(); i++) {
            ConceptRelation rel = relations.get(i);
            addEdge(rel.getSource(), rel.getTarget(), rel);
        }

        for (int i = 0; i < nodes.size(); i++) {
            getOrAddNode(nodes.get(i));
        }
    }

    public int getOutDegree(Concept node) {
        return this.kg.getOutDegree(node);
    }

    public List<ConceptRelation> getOutgoingEdges(Concept node) {
        return this.kg.getOutgoingEdges(node);
    }

    public int getInDegree(Concept node) {
        return this.kg.getInDegree(node);
    }

    public List<ConceptRelation> getIncomingEdges(Concept node) {
        return this.kg.getIncomingEdges(node);
    }

    public List<ConceptRelation> getEdges(Concept source, Concept dest) {
        return this.kg.getEdges(source, dest);
    }

    //merge coreferent concept nodes in the knowledge graph
    public void mergeCoreferentNodes()
    {
    	//to do
    }

}
