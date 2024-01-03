/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.model;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

import bass.data.ConceptSource;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.ErasureUtils;

public class KnowledgeGraphCoreAnnotations {
	
	private KnowledgeGraphCoreAnnotations() {}
	
	public static class IndicesAnnotation implements CoreAnnotation<Set<Integer>>, Serializable {

		private static final long serialVersionUID = -8586412448341997431L;
	
		public Class<Set<Integer>> getType() {
	        return ErasureUtils.<Class<Set<Integer>>> uncheckedCast(Set.class);
	    }
	}

	public static class ConceptSourcesAnnotation implements CoreAnnotation<Set<ConceptSource>>, Serializable {
		private static final long serialVersionUID = -701016810922189957L;
	
		public Class<Set<ConceptSource>> getType() {
	        return ErasureUtils.<Class<Set<ConceptSource>>> uncheckedCast(Set.class);
	    }
	}

	public static class CompoundIndexedWordAnnotation implements CoreAnnotation<List<IndexedWord>>, Serializable {
		private static final long serialVersionUID = 5651302323594616798L;

		public Class<List<IndexedWord>> getType() {
	        return ErasureUtils.<Class<List<IndexedWord>>> uncheckedCast(List.class);
	    }
    }
	
    public static class CompoundWordAnnotation implements CoreAnnotation<String>, Serializable {
		private static final long serialVersionUID = -8852477606188450189L;
	
		public Class<String> getType() {
	        return String.class;
	    }
    }

    public static class CompoundLemmaAnnotation implements CoreAnnotation<String>, Serializable {
		private static final long serialVersionUID = 2596178101491628802L;
	
		public Class<String> getType() {
	        return String.class;
	    }
    }

    public static class ConceptHeadAnnotation implements CoreAnnotation<Boolean>, Serializable {
		private static final long serialVersionUID = 6278961712646206186L;
	
		public Class<Boolean> getType() {
	        return Boolean.class;
	    }
	}
}