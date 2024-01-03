/**
 * Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
 * Author: Osman Alperen Koras
 * Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
 * Email: osman.koras@uni-due.de
 * Date: 12.10.2023
 *
 * License: MIT
 */
 package helpers;

import preprocess.SummarizationDatapoint;

import javax.json.Json;
import javax.json.JsonObject;
import java.io.*;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class FileIO {
    /**
     * Deletes the contents of the specified directory.
     * @param temp_dir The directory to be deleted.
     * @throws IOException
     */
    public static void delete_directory(Path temp_dir) throws IOException {
        try (var dirStream = Files.walk(temp_dir)) {
            dirStream
                    .map(Path::toFile)
                    .sorted(Comparator.reverseOrder())
                    .forEach(File::delete);
        }
    }

    /**
     * Reads from the given GZIP-compressed file (BigPatent dataset file) and parses documents into our
     * SummarizationDatapoint structure.
     * @param dir The directory containing the file.
     * @param file_path The path to the file.
     * @return A list of SummarizationDatapoints.
     * @throws IOException
     */
    public static ArrayList<SummarizationDatapoint> loadDatapoints(Path dir, Path file_path) throws IOException {
        var datapoints = new ArrayList<SummarizationDatapoint>();
        var file = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(dir.resolve(file_path).toString()))));
        // use the line below for WikiSumDataset.jsonl
        // var file = new BufferedReader(new InputStreamReader(new FileInputStream(dir.resolve(file_path).toString())));

        int i = 0;
        while(true) {
            var line = file.readLine();
            if (line == null)
                break;
            var reader = Json.createReader(new StringReader(line));
            JsonObject obj = reader.readObject();

            var datapoint = new SummarizationDatapoint();
            datapoint.id = file_path + ":" + i;
            datapoint.document_text = obj.getString("description");
            datapoint.summary_text = obj.getString("abstract");
            // Use the lines below for WikiSumDataset
            // datapoint.document_text = obj.getString("article");
            // datapoint.summary_text = obj.getString("summary");
            datapoints.add(datapoint);

            i++;
        }
        return datapoints;
    }

    /**
     * Copies individual files from the source directory to the target directory.
     * @param source_dir The source directory.
     * @param target_dir The target directory.
     * @param files The list of files to be copied.
     * @throws IOException
     */
    public static void copy_files(Path source_dir, Path target_dir, ArrayList<Path> files) throws IOException {
        for (var file: files) {
            var source = source_dir.resolve(file);
            if(Files.isRegularFile(source)) {
                var destination = target_dir.resolve(file);
                Files.createDirectories(destination.getParent());
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    /**
     * Copies directories from the source directory to the target directory.
     * @param source_dir The source directory.
     * @param target_dir The target directory.
     * @param directories The list of directories to be copied.
     * @throws IOException
     */
    public static void copy_directories(Path source_dir, Path target_dir, ArrayList<Path> directories) throws IOException {
        for (var directory: directories) {
            var source = source_dir.resolve(directory);
            if(Files.isDirectory(source)) {
                var destination = target_dir.resolve(directory);
                Files.createDirectories(destination);
                try (Stream<Path> files = Files.walk(source)) {
                    for (var file : files.toList()) {
                        if (file.toString().matches(".*\\.gz/[0-9]+$")) {
                            Files.copy(source.resolve(file.getFileName()), destination.resolve(file.getFileName()), StandardCopyOption.REPLACE_EXISTING);
                        }
                    }
                }
            }
        }
    }
}
