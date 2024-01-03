/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;
import java.util.List;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.ArrayCoreMap;
import bass.model.KnowledgeGraphCoreAnnotations;

/*
 * Represent a concept, which contains a noun phrase and its entity type
 * @author Wei Li
 */

public class Concept extends ArrayCoreMap implements Comparable<Concept>, Serializable {

    private static final long serialVersionUID = 5877090816262265315L;
    private List<IndexedWord> conceptTokens;
    private String conceptType;

    public Concept() {}

    public Concept(List<IndexedWord> conceptTokens, String conceptType) {
        this.conceptTokens = conceptTokens;
        this.conceptType = conceptType;
    }
    
    public IndexedWord getConceptHead() {
        if (conceptTokens == null || conceptTokens.isEmpty()) {
            System.out.println("Concept.java 28: concept doesn't contain tokens!!!");
            return null;
        }

        for (int i = conceptTokens.size() - 1; i >= 0; i--) {
            IndexedWord token = conceptTokens.get(i);
            if (token.containsKey(KnowledgeGraphCoreAnnotations.ConceptHeadAnnotation.class)
                    && token.get(KnowledgeGraphCoreAnnotations.ConceptHeadAnnotation.class)) {
                return token;
            }
        }
        IndexedWord token = conceptTokens.get(conceptTokens.size() - 1);
        token.set(KnowledgeGraphCoreAnnotations.ConceptHeadAnnotation.class, true);
        return token;
    }

    public void setConceptTokens(List<IndexedWord> conceptTokens) {
        this.conceptTokens = conceptTokens;
    }

    public List<IndexedWord> getConceptTokens() {
        return conceptTokens;
    }

    public void setConceptType(String conceptType) {
        this.conceptType = conceptType;
    }

    public String getConceptType() {
        return conceptType;
    }

    /*
     * Sort concepts according to doc_id, sen_id and index of token in the sentence*/
    @Override
    public int compareTo(Concept o) {
        if (o == null) {
            return 1;
        }
        int index1 = this.getConceptHead().index();
        int senInd1 = this.getConceptHead().sentIndex();
        int docInd1 = Integer.parseInt(this.getConceptHead().docID());

        int index2 = o.getConceptHead().index();
        int senInd2 = o.getConceptHead().sentIndex();
        int docInd2 = Integer.parseInt(o.getConceptHead().docID());

        if (docInd1 < docInd2)
            return -1;
        else if (docInd1 > docInd2)
            return 1;
        else {
            if (senInd1 < senInd2)
                return -1;
            else if (senInd1 > senInd2)
                return 1;
            else {
                if (index1 < index2)
                    return -1;
                else if (index1 > index2)
                    return 1;
                else
                    return 0;
            }

        }
    }

    @Override
    public String toString() {
        String conStr = conceptType + ":\n";
        for (IndexedWord token : conceptTokens) {
            //conStr += token.toString() + " ";
            conStr += token.docID()+":"+token.sentIndex()+":"+token.index()+":"+token.lemma()+" ";
        }
        conStr += "\n";
        conStr += "------------------------\n";
        return conStr;
    }

    public String tokenString() {
        String tokenStr = "";
        for (IndexedWord token : conceptTokens) {
            tokenStr += token.word() + " ";
        }
        return tokenStr;
    }

    public void copy(Concept o) {
        this.conceptTokens = o.getConceptTokens();
        this.conceptType = o.getConceptType();
    }

    /* (non-Javadoc)
     * @see java.lang.Object#hashCode()
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;//super.hashCode();
        result = prime * result
                + ((conceptTokens == null) ? 0 : conceptTokens.hashCode());
        result = prime * result + ((conceptType == null) ? 0 : conceptType.hashCode());
        return result;
    }

    /* (non-Javadoc)
     * @see java.lang.Object#equals(java.lang.Object)
     */
    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }

        if (!(other instanceof Concept)) {
            return false;
        }
        
        Concept other_obj = (Concept) other;
        if (conceptTokens == null) {
            if (other_obj.conceptTokens != null) {
                return false;
            }
        } else if (!this.getConceptHead().equals(other_obj.getConceptHead())) {
            return false;
        }
        
        return true;
    }

    public boolean stringMatch(String objStr) {
        String conStr = this.tokenString().trim();
        if (objStr.trim().equals(conStr))
            return true;
        else
            return false;
    }
}
