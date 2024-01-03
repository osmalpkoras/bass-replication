/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.HashMap;


import edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.util.CoreMap;


public class Document implements Serializable {

	private static final long serialVersionUID = -2900241239536019472L;
	public int docNo;
	public List<String> sentences_list;
	private List<Sentence> sentences;
	private List<CorefChain> corefChainList;// the coreferences contained in the doucument
	private List<Coreference> coreferences;
	private KnowledgeGraph doc_kg;
	private SemanticGraph doc_sg;
	public Map<String,List<List<Integer>>> SentenceProj;
	public List<String> RootNodes;

	public Document() {
		// TODO Auto-generated constructor stub
		docNo = 0;
		sentences = new ArrayList<>();
		corefChainList = new ArrayList<>();
		coreferences = new ArrayList<>();
		doc_kg = null;
		doc_sg = null;
		SentenceProj = new HashMap<String,List<List<Integer>>>();
		RootNodes = new ArrayList<String>();
	}

	public Document(int docNo, String doc_content, StanfordCoreNLP pipeline) {
		this.docNo = docNo;
		sentences_list = new ArrayList<String>();
		//sentences_list = Arrays.asList(doc_content.split(" . "));
		Annotation document = new Annotation(doc_content);
		pipeline.annotate(document);

		int senInd = 0;
		sentences = new ArrayList<>();
		List<CoreMap> sentenceAnnotation = document.get(SentencesAnnotation.class);
		for (CoreMap sen : sentenceAnnotation) {
			Sentence s = new Sentence(docNo, senInd, sen);
			sentences_list.add(s.getRawtext());
			//sentences_list.add(s.getTokens());
			sentences.add(s);
			senInd++;
		}

		corefChainList = new ArrayList<>();
		SentenceProj = new HashMap<String,List<List<Integer>>>();
		RootNodes = new ArrayList<String>();
		Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
		Iterator<Integer> iter = graph.keySet().iterator();
		while (iter.hasNext()) {
			Integer key = iter.next();
			CorefChain coref = graph.get(key);
			corefChainList.add(coref);
		}
		constructCorferences();
	}

	public Document(int docNo, Annotation document) {
		this.docNo = docNo;
		sentences_list = new ArrayList<String>();
		//sentences_list = Arrays.asList(doc_content.split(" . "));

		int senInd = 0;
		sentences = new ArrayList<>();
		List<CoreMap> sentenceAnnotation = document.get(SentencesAnnotation.class);
		for (CoreMap sen : sentenceAnnotation) {
			Sentence s = new Sentence(docNo, senInd, sen);
			sentences_list.add(s.getRawtext());
			//sentences_list.add(s.getTokens());
			sentences.add(s);
			senInd++;
		}

		corefChainList = new ArrayList<>();
		SentenceProj = new HashMap<String,List<List<Integer>>>();
		RootNodes = new ArrayList<String>();
		Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
		Iterator<Integer> iter = graph.keySet().iterator();
		while (iter.hasNext()) {
			Integer key = iter.next();
			CorefChain coref = graph.get(key);
			corefChainList.add(coref);
		}
		constructCorferences();
	}

	public Document(int docNo, File doc_file, StanfordCoreNLP pipeline) {
		Annotation document = null;
		try {
			document = new Annotation(IOUtils.slurpFile(doc_file));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		pipeline.annotate(document);

		int senInd = 0;
		sentences = new ArrayList<>();
		SentenceProj = new HashMap<String,List<List<Integer>>>();
		RootNodes = new ArrayList<String>();
		List<CoreMap> sentenceAnnotation = document.get(SentencesAnnotation.class);
		for (CoreMap sen : sentenceAnnotation) {
			Sentence s = new Sentence(docNo, senInd, sen);
			sentences.add(s);
			senInd++;
		}

		corefChainList = new ArrayList<>();
		Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
		Iterator<Integer> iter = graph.keySet().iterator();
		while (iter.hasNext()) {
			Integer key = iter.next();
			CorefChain coref = graph.get(key);
			corefChainList.add(coref);
		}
		constructCorferences();
	}

	public void setDocNo(int docNo) {
		this.docNo = docNo;
	}

	public int getDocNo() {
		return docNo;
	}

	public void setSentences(ArrayList<Sentence> sentences) {
		this.sentences = sentences;
	}

	public List<Sentence> getSentences() {
		return sentences;
	}

	public void setCorefChainList(List<CorefChain> corefChainList) {
		this.corefChainList = corefChainList;
	}

	public List<CorefChain> getCorefChainList() {
		return corefChainList;
	}

	public KnowledgeGraph getKnowledgeGraph() {
		return this.doc_kg;
	}

	public void setKnowledgeGraph(KnowledgeGraph kg) {
		this.doc_kg = kg;
	}

	public SemanticGraph getSemanticGraph() {
		return this.doc_sg;
	}

	public void setSemanticGraph(SemanticGraph sg) {
		this.doc_sg = sg;
	}

	public void constructCorferences() {
		coreferences = new ArrayList<>();
		
		for (CorefChain corefChain : corefChainList) {
			Coreference coref = new Coreference();
			List<CorefMention> mentions = corefChain.getMentionsInTextualOrder();
			//System.out.println(mentions);
			int first = 1;
			for (CorefMention mention : mentions) {
				int representative=mention.mentionType.representativeness;
				int senIdx = mention.sentNum - 1;
				int startIdx = mention.startIndex - 1;
				int endIdx = mention.endIndex - 1;
				int headIdx = mention.headIndex - 1;
				Mention ment;
				if (first == 1) {
					ment = new Mention(true, senIdx, startIdx, endIdx, headIdx);
					first = 0;
				} else
					ment = new Mention(false, senIdx, startIdx, endIdx, headIdx);
				coref.addMention(ment);
			}
			coreferences.add(coref);
		}


	}

	public void setCoreferences(ArrayList<Coreference> coreferences) {
		this.coreferences = coreferences;
	}

	public List<Coreference> getCoreferences() {
		return coreferences;
	}

	public String getPhraseRepresentative(int senNo, int headIdx) {
		List<Coreference> corefs = this.coreferences;
		
		String repStr = "";
		for (int i = 0; i < corefs.size(); i++) {
			Coreference cof = corefs.get(i);
			ArrayList<Mention> mentions = cof.getMentions();
			for (int j = 0; j < mentions.size(); j++) {
				Mention mention = mentions.get(j);
				if (mention.getRepresentative() == false && mention.getSentence_idx() == senNo
						&& mention.getHead_idx() == headIdx) {
					int rep_SenNo = mentions.get(0).getSentence_idx();
					int startIdx = mentions.get(0).getWord_start_idx();
					int endIdx = mentions.get(0).getWord_end_idx();

					Sentence repSen = sentences.get(rep_SenNo);
					List<IndexedWord> tokens = repSen.getTokens();
					for (int k = startIdx; k < endIdx; k++) {
						if (repStr.equalsIgnoreCase(""))
							repStr = tokens.get(k).lemma();
						else
							repStr += " " + tokens.get(k).lemma();
					}
					break;
				}
			}
			if (!repStr.equalsIgnoreCase(""))
				break;
		}
		return repStr;
	}

	public ArrayList<IndexedWord> getArrayListPhraseRepresentative(int senNo, int headIdx) {
		List<Coreference> corefs = this.coreferences;
		
		ArrayList<IndexedWord> repStr = new ArrayList<>();
		for (int i = 0; i < corefs.size(); i++) {
			Coreference cof = corefs.get(i);
			ArrayList<Mention> mentions = cof.getMentions();
			for (int j = 0; j < mentions.size(); j++) {
				Mention mention = mentions.get(j);
				if (mention.getRepresentative() == false && mention.getSentence_idx() == senNo
						&& mention.getHead_idx() == headIdx) {
					int rep_SenNo = mentions.get(0).getSentence_idx();
					int startIdx = mentions.get(0).getWord_start_idx();
					int endIdx = mentions.get(0).getWord_end_idx();

					Sentence repSen = sentences.get(rep_SenNo);
					List<IndexedWord> tokens = repSen.getTokens();
					for (int k = startIdx; k < endIdx; k++) {
						repStr.add(tokens.get(k));
					}
					break;
				}
			}
			if (!repStr.isEmpty())
				break;
		}
		return repStr;
	}

	/*
	 * get the representative phrase for the token (senNo, headIdx) if the token is
	 * representative, return null
	 */
	public Mention getMentionPhraseRepresentative(int senNo, int headIdx) {
		List<Coreference> corefs = this.coreferences;
		for (int i = 0; i < corefs.size(); i++) {
			Coreference cof = corefs.get(i);
			ArrayList<Mention> mentions = cof.getMentions();
			for (int j = 0; j < mentions.size(); j++) {
				Mention mention = mentions.get(j);
				if (mention.getRepresentative() == false && mention.getSentence_idx() == senNo
						&& mention.getHead_idx() == headIdx) {
					cof.visited = true;
					return mentions.get(0);
				}
			}
		}
		return null;
	}
}
