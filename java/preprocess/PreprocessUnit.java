/**
 * Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
 * Author: Osman Alperen Koras
 * Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
 * Email: osman.koras@uni-due.de
 * Date: 12.10.2023
 *
 * License: MIT
 */
 package preprocess;

import bass.model.SemanticGraphParser;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import semanticgraph.Document;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.concurrent.Callable;

/**
 * This class represents a unit of pre-processing work that processes a single chunk using Stanford CoreNLP.
 * It annotates each chunk with linguistic information, and constructs a semantic graph for the chunk.
 */
public class PreprocessUnit implements Callable<Void> {
    private final StanfordCoreNLP pipeline;
    private final ArrayList<String> inputs;
    private final ArrayList<Document> docs;
    private final ProtobufAnnotationSerializer serializer;
    private final ArrayList<byte[]> protobuf_annotation;
    private final int i;

    /**
     * Constructs a preprocess.PreprocessUnit.
     *
     * @param pipeline           The Stanford CoreNLP pipeline for annotating text.
     * @param inputs             A list of input strings, each representing a chunk of text.
     * @param docs               A list to store Document objects representing semantic graphs.
     * @param serializer         A ProtobufAnnotationSerializer for serializing CoreNLP annotations.
     * @param protobufAnnotation A list to store serialized CoreNLP annotations.
     * @param i                  The index of the chunk to be pre-processed.
     */
    public PreprocessUnit(StanfordCoreNLP pipeline, ArrayList<String> inputs, ArrayList<Document> docs, ProtobufAnnotationSerializer serializer, ArrayList<byte[]> protobufAnnotation, int i) {
        this.pipeline = pipeline;
        this.inputs = inputs;
        this.docs = docs;
        this.serializer = serializer;
        this.protobuf_annotation = protobufAnnotation;
        this.i = i;
    }

    @Override
    public Void call() throws Exception {
        // Annotate the input string with linguistic information using Stanford CoreNLP.
        var doc = new CoreDocument(inputs.get(i));
        pipeline.annotate(doc);

        // Serialize the CoreNLP annotations for this chunk and add it to the protobuf_annotation list.
        var out = new ByteArrayOutputStream();
        serializer.writeCoreDocument(doc, out);
        protobuf_annotation.add(out.toByteArray());

        // Parse the semantic graph for the chunk and store it in the docs list.
        var bassDoc = new bass.data.Document(i, doc.annotation());
        (new SemanticGraphParser()).parse(bassDoc);
        docs.add(new Document(bassDoc, doc.text()));
        return null;
    }
}
