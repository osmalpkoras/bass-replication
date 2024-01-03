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
import java.io.ObjectInputStream;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

public class FileInputStream<TData> implements AutoCloseable {
    private final ObjectInputStream stream;

    public FileInputStream(Path filepath) throws IOException {
        this.stream = new ObjectInputStream(new GZIPInputStream(new java.io.FileInputStream(filepath.toString())));
    }

    public TData read() throws IOException, ClassNotFoundException {
        return (TData) this.stream.readObject();
    }

    @Override
    public void close() throws Exception {
        this.stream.close();
    }
}
