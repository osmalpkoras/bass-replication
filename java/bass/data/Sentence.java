/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class Sentence implements Serializable{
	private static final long serialVersionUID = -2380552389809944654L;
	private int docNo;
	private int senInd; //the position in the document
	private String rawText;
	private List<IndexedWord> tokens;
	private List<CoreMap> entityMentions;
	private List<Concept> concepts;
	private KnowledgeGraph kg;//the knowledge graph from the sentence
	private SemanticGraph sg;//the dependency parser of the sentence
	private int letterLength;

	public Sentence() {
		docNo = -1;
		senInd = -1;
	}

	public Sentence(int docNo, int senInd, CoreMap sen)
	{
		// TODO Auto-generated constructor stub
		this.docNo = docNo;
		this.senInd = senInd;

		rawText = sen.get(CoreAnnotations.TextAnnotation.class);
		sg = sen.get(EnhancedPlusPlusDependenciesAnnotation.class);
		entityMentions = sen.get(CoreAnnotations.MentionsAnnotation.class);

		tokens = sg.vertexListSorted();
		int index = 0;
		for (IndexedWord token: tokens)
		{
			token.set(CoreAnnotations.DocIDAnnotation.class, docNo+"");
			token.set(CoreAnnotations.SentenceIndexAnnotation.class, senInd);
			token.set(CoreAnnotations.IndexAnnotation.class, index);
			index ++;
		}
		letterLength = tokens.get(index-1).endPosition()+1;
	}

	public void setDocNo( int docNo)
	{
		this.docNo = docNo;
	}

	public int getDocNo()
	{
		return docNo;
	}

	public void setSenInd( int senInd)
	{
		this.senInd = senInd;
	}

	public int getSenInd()
	{
		return senInd;
	}

	public void setRawtext( String rawText)
	{
		this.rawText = rawText;
	}

	public String getRawtext()
	{
		return rawText;
	}

	public void setTokens( List<IndexedWord> tokens)
	{
		this.tokens = tokens;
	}

	public List<IndexedWord> getTokens()
	{
		return tokens;
	}

	public void setEntityMentions(List<CoreMap> entityMentions)
	{
		this.entityMentions = entityMentions;
	}

	public List<CoreMap> getEntityMentions()
	{
		return entityMentions;
	}

	public void setConcepts(List<Concept> concepts)
	{
		this.concepts = concepts;
	}

	public List<Concept> getConcepts()
	{
		return concepts;
	}

	public void setKnowledgeGraph(KnowledgeGraph kg)
	{
		this.kg = kg;
	}

	public KnowledgeGraph getKnowledgeGraph()
	{
		return kg;
	}

	public void setSemanticGraph(SemanticGraph sg)
	{
		this.sg = sg;
	}

	public SemanticGraph getSemanticGraph()
	{
		return sg;
	}

	public int getLetterLength()
	{
		return letterLength;
	}
}
