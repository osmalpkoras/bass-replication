/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;
import java.util.List;


import edu.stanford.nlp.ling.IndexedWord;

/*
 * The relation between concepts*/

public class ConceptRelation implements Comparable<ConceptRelation>, Serializable{

	private static final long serialVersionUID = -2507217320171623112L;
	private Concept source;
	private Concept target;
	private List<IndexedWord> relationTokens;

	public ConceptRelation(Concept source, Concept target,List<IndexedWord> relation) {
		this.source = source;
		this.target = target;
		this.relationTokens = relation;
	}

	public Concept getSource() {
		return source;
	}

	public Concept getTarget() {
		return target;
	}

	public List<IndexedWord> getRelationTokens() {
		return relationTokens;
	}

	public String getRelationString()
	{
		String rel_str = "";
		for(int i = 0; i < relationTokens.size(); i++)
		{
			rel_str += relationTokens.get(i).word();
			if(i < relationTokens.size() - 1)
				rel_str += " ";
		}
		
		return rel_str;
	}

	@Override
	public int compareTo(ConceptRelation o) {
		if (o == null) {
			return 1;
		}

		int ret = this.source.compareTo(o.source);
		if (ret != 0) {
			return ret;
		}

		ret = this.target.compareTo(o.target);

		if (ret != 0) {
			return ret;
		}

		ret = this.relationTokens.get(0).compareTo(o.relationTokens.get(0));

		return ret;
	}

	@Override
	public int hashCode() {
		int result = 0;
		result += 29*this.source.hashCode();
		result += 29*this.target.hashCode();
		result += 29*this.relationTokens.get(0).hashCode();
		return result;
	}

	@Override
	public boolean equals(Object o) {
		if (o == null) {
			return false;
		}

		if (!(o instanceof ConceptRelation)) {
			return false;
		}

		ConceptRelation oReln = (ConceptRelation) o;

		return this.source.equals(oReln.source)
				&& this.target.equals(oReln.target)
				&& this.relationTokens.get(0).equals(oReln.relationTokens.get(0));
	}
	
	
	public String toString() {
		String attr = this.getRelationString() + "-->" + target.tokenString(); 
		return attr; 
	} 
}
