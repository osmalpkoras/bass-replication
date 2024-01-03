/**
 * Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
 * Author: Osman Alperen Koras
 * Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
 * Email: osman.koras@uni-due.de
 * Date: 12.10.2023
 *
 * License: MIT
 */
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import helpers.FileIO;
import preprocess.CoreNlpDatapoint;
import preprocess.PreprocessUnit;
import preprocess.SummarizationDatapoint;
import semanticgraph.Document;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Stream;


import static java.lang.Integer.min;

public class Preprocessor {
    public static void main(String[] args) throws Exception {
        var m = new Preprocessor();
        m.start_preprocessing(
                Paths.get(args[0]).toAbsolutePath().normalize(),
                Paths.get(args[1]).toAbsolutePath().normalize(),
                Paths.get(args[2]).toAbsolutePath().normalize(),
                Integer.parseInt(args[3]),
                Integer.parseInt(args[4]),
                Integer.parseInt(args[5]),
                Integer.parseInt(args[6]),
                Integer.parseInt(args[7]));
    }

    /**
     * Main entry point to start pre-processing. After pre-processing with CoreNLP, the original source code for graph
     * construction will also be called. Pre-processing output and graph construction output are saved alongside each
     * other.
     * @param source_dir The directory containing the extracted BigPatent zip archive.
     * @param temp_dir A temporary directory used for intermediate files. All input files are copied into this directory. Output files are generated inside this directory. For jobs running in a cluster, pick a local folder. This directory will be cleaned up afterwards.
     * @param target_dir The target directory into which all output files will be copied to.
     * @param pool_index The index of the worker pre-processing this dataset (0 <= pool_index <= pool_size).
     * @param pool_size The number of workers separately pre-processing this dataset.
     * @param timeout The time in seconds given to pre-process a chunk
     * @param chunk_size The number of words (strings delimited by whitespace) per chunk. Each chunk is cut off after the next sentence stop.
     * @param max_chunks The maximum number of chunks to pre-process.
     * @throws Exception
     */
    private void start_preprocessing(Path source_dir, Path temp_dir, Path target_dir, int pool_index, int pool_size, int timeout, int chunk_size, int max_chunks) throws Exception {
        System.out.println("source_dir: " + source_dir);
        System.out.println("temp_dir: " + temp_dir);
        System.out.println("target_dir: " + target_dir);
        System.out.println("pool_index: " + pool_index);
        System.out.println("pool_size: " + pool_size);
        System.out.println("timeout: " + timeout);
        System.out.println("chunk_size: " + chunk_size);
        System.out.println("max_chunks: " + max_chunks);
        System.out.println("==========================");

        // create temporary directory structure
        UUID sub = generate_unique_subdirectory(temp_dir);
        temp_dir = temp_dir.resolve(sub.toString());
        var input_dir = temp_dir.resolve("input");
        var output_dir = temp_dir.resolve("output");

        Files.createDirectories(target_dir);
        Files.createDirectories(input_dir);
        Files.createDirectories(output_dir);
        System.out.println("directory structure created...");

        // get files to be pre-processed by this worker
        ArrayList<Path> files = getFiles(source_dir, pool_index, pool_size);
        System.out.println("dataset files identified...");

        var startTime = System.nanoTime();

        // copy source files into the temporary folder
        FileIO.copy_files(source_dir, input_dir, files);
        System.out.println("dataset files copied to temporary folder...");

        // pre-process documents
        run_preprocessing(target_dir, input_dir, output_dir, files, timeout, chunk_size, max_chunks);

        var endTime = System.nanoTime();
        System.out.println("preprocessing finished in "+(endTime - startTime) / 1000000000.);

        // copy_files(output_dir, target_dir, files);
        FileIO.copy_directories(output_dir, target_dir, files);
        System.out.println("now copied files to target_dir...");

        FileIO.delete_directory(temp_dir);
        System.out.println("now cleaned up temp_dir...");
    }

    /**
     * Retrieves a list of files to be pre-processed by the current worker.
     * @param source_dir The source directory containing the dataset.
     * @param pool_index The index of the worker.
     * @param pool_size The total number of workers.
     * @return A list of files for the current worker.
     */
    private ArrayList<Path> getFiles(Path source_dir, int pool_index, int pool_size) {
        var allFiles = new ArrayList<Path>();
        for (var t : new String[]{"test", "val", "train"}) {
            if(!Files.isDirectory(source_dir.resolve(t))) {
                continue;
            }
            var sub_files = new ArrayList<Path>();
            try (Stream<Path> paths = Files.walk(source_dir.resolve(t))) {
                for (var path : paths.toList()) {
                    if (path.toString().matches(".*\\.gz$")) {
                        sub_files.add(source_dir.relativize(path));
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            sub_files.sort(Comparator.comparing(Path::toString));
            allFiles.addAll(sub_files);
        }

        ArrayList<Path> files = new ArrayList<Path>();
        for (int i = pool_index; i < allFiles.size(); i = i + pool_size) {
            files.add(allFiles.get(i));
        }
        return files;
    }

    /**
     * Generates a unique subdirectory under the given temporary directory.
     * @param temp_dir The temporary directory.
     * @return A unique subdirectory name.
     */
    private UUID generate_unique_subdirectory(Path temp_dir) {
        var sub = UUID.randomUUID();
        while (Files.exists(temp_dir.resolve(sub.toString()))) {
            sub = UUID.randomUUID();
        }
        return sub;
    }

    /**
     * Pre-processes documents using Stanford CoreNLP.
     * @param target_dir The target directory where pre-processed files will be saved.
     * @param input_dir The input directory containing source files.
     * @param output_dir The output directory for pre-processed files.
     * @param files The list of files to be processed by the current worker.
     * @param timeout The timeout for pre-processing a chunk
     * @param chunk_size The size of document chunks.
     * @param max_chunks The maximum number of chunks to pre-process.
     * @throws Exception
     */
    private void run_preprocessing(Path target_dir, Path input_dir, Path output_dir, ArrayList<Path> files, int timeout, int chunk_size, int max_chunks) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        for (Path file_path : files) {
            try {
                System.out.println("now processing " + file_path + "...");
                Files.createDirectories(output_dir.resolve(file_path));
                ArrayList<SummarizationDatapoint> datapoints = FileIO.loadDatapoints(input_dir, file_path);
                System.out.println(datapoints.size() + " source datapoints loaded...");
                StanfordCoreNLP pipeline = getCoreNlpPipeline(chunk_size);

                for (var datapoint : datapoints) {
                    try {
                        var id = datapoint.id.split(":")[1];
                        if (Files.exists(target_dir.resolve("stop_preprocessing"))) {
                            System.out.println("found a break file... stop preprocessing");
                            return;
                        }

                        if (Files.exists(target_dir.resolve(file_path).resolve(id)))
                            continue;

                        System.out.println("now preprocessing " + datapoint.id);
                        ArrayList<String> inputs = chunkString(datapoint.document_text, chunk_size, max_chunks);
                        System.out.println(inputs.size() + " chunks created...");
                        var docs = new ArrayList<Document>();
                        ProtobufAnnotationSerializer serializer = new ProtobufAnnotationSerializer();
                        var protobuf_annotation = new ArrayList<byte[]>();

                        for (int i = 0; i < inputs.size(); i++) {
                            if (Files.exists(target_dir.resolve("stop_preprocessing"))){
                                System.out.println("found a break file... stop preprocessing");
                                System.out.println("failed to preprocess " + datapoint.id);
                                return;
                            }

                            Future<?> future = executor.submit(new PreprocessUnit(pipeline, inputs, docs, serializer, protobuf_annotation, i));
                            try {
                                future.get(timeout, TimeUnit.SECONDS);
                            } catch (TimeoutException e) {
                                future.cancel(true);
                                throw e;
                            }
                        }

                        var out_datapoint = new CoreNlpDatapoint();
                        out_datapoint.id = datapoint.id;
                        out_datapoint.document_text = datapoint.document_text;
                        out_datapoint.summary_text = datapoint.summary_text;
                        out_datapoint.protobuf_annotation = protobuf_annotation;
                        out_datapoint.annotation = docs;

                        try (var output_file = new helpers.FileOutputStream<CoreNlpDatapoint>(output_dir.resolve(file_path).resolve(id))) {
                            output_file.write(out_datapoint);
                        }
                        catch (WriteAbortedException ex) {
                            Files.deleteIfExists(output_dir.resolve(file_path).resolve(id));
                            throw ex;
                        }
                    } catch (Throwable ex) {
                        ex.printStackTrace();
                        System.err.println("failed to preprocess " + datapoint.id);
                        System.out.println("failed to preprocess " + datapoint.id);
                    }
                }
            }
            catch (Throwable ex) {
                ex.printStackTrace();
                System.err.println("failed to preprocess");
                System.out.println("failed to preprocess");
            }
        }
        shutdownAndAwaitTermination(executor);
    }

    /**
     * Shuts down the executor service and waits for termination.
     * @param pool The executor service to be shut down.
     */
    void shutdownAndAwaitTermination(ExecutorService pool) {
        pool.shutdown(); // Disable new tasks from being submitted
        try {
            // Wait a while for existing tasks to terminate
            if (!pool.awaitTermination(60, TimeUnit.SECONDS)) {
                pool.shutdownNow(); // Cancel currently executing tasks
                // Wait a while for tasks to respond to being cancelled
                if (!pool.awaitTermination(60, TimeUnit.SECONDS))
                    System.err.println("Pool did not terminate");
            }
        } catch (InterruptedException ie) {
            // (Re-)Cancel if current thread also interrupted
            pool.shutdownNow();
            // Preserve interrupt status
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Creates a Stanford CoreNLP pipeline with specified annotators.
     * @param chunk_size The chunk size.
     * @return A Stanford CoreNLP pipeline.
     */
    private StanfordCoreNLP getCoreNlpPipeline(int chunk_size) {
        Properties props_nlp = new Properties();
        props_nlp.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, coref");
        props_nlp.setProperty("depparse.extradependencies", "MAXIMAL");
        props_nlp.setProperty("coref.algorithm", "neural");
        if (chunk_size == -1) {
            props_nlp.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger");
            props_nlp.setProperty("parse.model", "edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz");
            props_nlp.setProperty("ner.model","edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz");
        }
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props_nlp);
        return pipeline;
    }

    /**
     * Splits a given input string into chunks based on word count and sentence stops.
     * @param input The input string to be split.
     * @param chunk_size The size of each chunk in terms of word count.
     * @param max_chunks The maximum number of chunks to create.
     * @return A list of chunks.
     */
    private static ArrayList<String> chunkString(String input, int chunk_size, int max_chunks) {
        int chk_size = chunk_size - 10;
        if (chunk_size == -1) {
            chk_size = Integer.MAX_VALUE;
        }
        var inputs = new ArrayList<String>();
        List<String> words = new ArrayList<>(Arrays.asList(input.split(" ")));
        while (!words.isEmpty()) {
            var block = new ArrayList<>(words.subList(0, min(chk_size, words.size())));
            words = new ArrayList<>(words.subList(min(chk_size, words.size()), words.size()));

            while (!words.isEmpty() && !block.get(block.size() - 1).equals(".")) {
                block.add(words.get(0));
                words = new ArrayList<>(words.subList(1, words.size()));
            }

            inputs.add(String.join(" ", block));
            if (inputs.size() == max_chunks) {
                break;
            }
        }
        return inputs;
    }

}