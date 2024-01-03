/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.BufferedWriter;

import java.io.InputStreamReader;

import java.io.IOException;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;

public class Corpus implements Serializable {
    private static final long serialVersionUID = -1711429530547048705L;
    private List<Document> documents;
    private KnowledgeGraph corpus_kg; // for MDS
    private SemanticGraph corpus_sg; // Collapsed SG
    public Map<String,List<List<Integer>>> SentenceProj;
    public List<String> RootNodes;
    public Corpus() {
        // TODO Auto-generated constructor stub
    	this.documents = new ArrayList<Document>();
    	this.corpus_kg = null;
    	this.corpus_sg = null;
    }

    public Corpus(List<Document> documents) {
        // TODO Auto-generated constructor stub
        this.documents = documents;
        this.corpus_kg = null;
        this.corpus_sg = null;
    }

    public Corpus(String DocSentences, StanfordCoreNLP pipeline,int ConcatLen)
            throws FileNotFoundException, UnsupportedEncodingException {
        // TODO Auto-generated constructor stub
    	this.documents = new ArrayList<Document>();
    	ConstructSingeDocument(DocSentences, pipeline);
        this.SentenceProj = new HashMap<String,List<List<Integer>>>();
        this.RootNodes = new ArrayList<String>();
        this.corpus_kg = null;
        this.corpus_sg = null;
    }

    public Corpus(String textPath, StanfordCoreNLP pipeline)
            throws FileNotFoundException, UnsupportedEncodingException {
        // TODO Auto-generated constructor stub
    	this.documents = new ArrayList<Document>();
        ConstructDocuments(textPath, pipeline);
        this.SentenceProj = new HashMap<String,List<List<Integer>>>();
        this.RootNodes = new ArrayList<String>();
        this.corpus_kg = null;
        this.corpus_sg = null;
    }

    public void setDocuments(List<Document> documents) {
        this.documents = documents;
    }

    public List<Document> getDocuments() {
        return this.documents;
    }

    public Document getDocument(int docNo) {
        for (Document doc : documents) {
            if (doc.getDocNo() == docNo)
                return doc;
        }
        return null;
    }

    public void setKnowledgeGraph(KnowledgeGraph kg) {
    	this.corpus_kg = kg;
    }

    public KnowledgeGraph getKnowledgeGraph() {
        return this.corpus_kg;
    }

    public void setSemanticGraph(SemanticGraph sg) {
    	this.corpus_sg = sg;
    }

    public SemanticGraph getSemanticGraph() {
        return this.corpus_sg;
    }
    public int ConstructSingeDocument(String DocSentences,StanfordCoreNLP pipeline) {

      //String input_sentences = String.join(" . ",DocSentences);

      //System.out.println(String.join(" ",input_sentences));

      //String [] concated_sentences = input_sentences.split(".");
      int Block_Size = 500;
      String buffstring = "";
      int BlockNo = 0;
      Document doc = new Document(BlockNo,DocSentences,pipeline);
      this.documents.add(doc);
      /*
      for(String s:DocSentences){
          //System.out.println(s);
          if(!buffstring.isEmpty()) buffstring += " . ";
          buffstring += s;

          if(buffstring.split(" ").length>=Block_Size-10){

              Document doc = new Document(BlockNo,buffstring,pipeline);
              this.documents.add(doc);
              BlockNo +=1;
              buffstring = "";
          }
      }
      if(!buffstring.equals("")){
        buffstring += " .";
     	 Document doc = new Document(BlockNo,buffstring,pipeline);
        this.documents.add(doc);
    }
    */
		  return 1;
    }
    public int ConstructDocuments(String textPath, StanfordCoreNLP pipeline)
            throws FileNotFoundException, UnsupportedEncodingException {
        File file = new File(textPath);
        if (!file.exists() || file.isHidden())
        	return -1;

        if (file.isDirectory()) {
            String[] filelist = file.list();
            int docNo = 0;
            for (int i = 0; i < filelist.length; i++) {
                String docPath = textPath + "/" + filelist[i];
                File docFile = new File(docPath);
                if (docFile.isHidden())
                    continue;
                Document doc = new Document(docNo, docFile, pipeline);
                documents.add(doc);
                docNo++;
            }
        } else {
        	try {
    			BufferedReader reader = IOUtils.readerFromString(textPath);
    			int docNo = 0;
    			for (String line = reader.readLine(); line != null; line = reader.readLine()) {
    				if (line.equals(""))
    					continue;
    				Document doc = new Document(docNo, line, pipeline);
    				this.documents.add(doc);
    			}
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
        }
        return 1;
    }

    public int size() {
        return this.documents.size();
    }

}
