/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;

public class ConceptSource implements Serializable{
	
	private static final long serialVersionUID = 3571652173153538170L;
	public int docId;
	public int senId;
	public int headWordId;
	public Concept concept;
	
	public ConceptSource() {
		// TODO Auto-generated constructor stub
	}

	public ConceptSource(int docId, int senId, int headWordId, Concept c) {
		// TODO Auto-generated constructor stub
		this.docId = docId;
		this.senId = senId;
		this.headWordId = headWordId;
		this.concept = new Concept();
		this.concept.copy(c);
	}
}