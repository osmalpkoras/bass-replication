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

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.WriteAbortedException;
import java.nio.file.*;
import java.util.zip.GZIPOutputStream;

public class FileOutputStream<TData> implements AutoCloseable {
    private ObjectOutputStream stream;
    private final Path filepath;

    public FileOutputStream(Path filepath, boolean overwrite) throws IOException {
        if(overwrite) {
            Files.deleteIfExists(filepath);
        }
        this.filepath = filepath;
    }

    public FileOutputStream(Path filepath) throws IOException {
        this(filepath, false);
    }

    public void write_save(TData object) throws IOException {
        var filepath_copy =  Paths.get(filepath.toString() + ".copy");
        if(Files.exists(filepath)) {
            Files.copy(filepath, filepath_copy, StandardCopyOption.REPLACE_EXISTING);
        }
        var append = Files.exists(filepath);
        var gzip = new GZIPOutputStream(new java.io.FileOutputStream(filepath_copy.toString(), append));
        try(var stream = append ? new AppendingObjectOutputStream(gzip) : new ObjectOutputStream(gzip)) {
            stream.writeObject(object);
            stream.reset();
        }
        catch (Throwable ex) {
            throw new WriteAbortedException("Failed to write object to this stream.", new Exception(ex));
        }

        Files.copy(filepath_copy, filepath, StandardCopyOption.REPLACE_EXISTING);
    }


    public void write(TData object) throws IOException {
        var append = Files.exists(filepath);
        var gzip = new GZIPOutputStream(new java.io.FileOutputStream(filepath.toString(), append));
        try(var stream = append ? new AppendingObjectOutputStream(gzip) : new ObjectOutputStream(gzip)) {
            stream.writeObject(object);
            stream.reset();
        }
        catch (Throwable ex) {
            throw new WriteAbortedException("Failed to write object to this stream.", new Exception(ex));
        }
    }

    @Override
    public void close() throws Exception {
        if(this.stream != null) {
            this.stream.close();
        }
    }
}
