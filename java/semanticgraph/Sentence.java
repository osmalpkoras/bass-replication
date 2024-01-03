/**
 * Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
 * Author: Osman Alperen Koras
 * Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
 * Email: osman.koras@uni-due.de
 * Date: 12.10.2023
 *
 * License: MIT
 */
package semanticgraph;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Sentence  implements Serializable {
    public Document document = null;
    public List<Token> tokens = new ArrayList<Token>();
    public Graph semantic_graph = null;

    @Override
    public String toString() {
        return tokens.stream().map(Object::toString).collect(Collectors.joining(" "));
    }
}
