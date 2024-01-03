/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.model;

import bass.data.Corpus;
import bass.data.Document;
import bass.data.Sentence;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
/**
 * A semantic graph parser.
 *
 * Takes a dependency parse, performs some enhancements using {@link SemanticGraphEnhancer},
 * Merging semantic graph of each sentence using {@link SemanticGraphParser::mergeSemanticGraph}
 * Merging coreferent node using {@link SemanticGraphParser::mergeCoreferentNodes}}.
 *
 * @author Wei Li
 */

public class SemanticGraphParser {

    public SemanticGraphParser() {

    }

    public void parse(Document doc)
    {
        SemanticGraphEnhancer.enhance(doc);
        SemanticGraph doc_sg = new SemanticGraph();
        for(Sentence sen: doc.getSentences())
        {
            SemanticGraph sen_sg = sen.getSemanticGraph();
            doc_sg = mergeSemanticGraph(doc_sg, sen_sg);
        }
        doc.setSemanticGraph(doc_sg);
    }

    public void parse(Corpus corpus)
    {
        SemanticGraph corpus_sg = new SemanticGraph();

        for (Document doc : corpus.getDocuments()) {
            parse(doc);
            SemanticGraph doc_sg = doc.getSemanticGraph();
            corpus_sg = mergeSemanticGraph(corpus_sg, doc_sg);
        }
        corpus.setSemanticGraph(corpus_sg);
    }

    /*merge two semantic graph*/
    public static SemanticGraph mergeSemanticGraph(SemanticGraph source, SemanticGraph target)
    {
        if (target == null)
            return source;

        for (SemanticGraphEdge rel: target.edgeListSorted())
            source.addEdge(rel);

        for (IndexedWord vertex: target.vertexListSorted())
            source.addVertex(vertex);

        for (IndexedWord root: target.getRoots())
            source.addRoot(root);

        return source;
    }
}
