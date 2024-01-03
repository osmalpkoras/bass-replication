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

public class Graph implements Serializable {

    public List<Node> nodes =  new ArrayList<Node>();
    public List<Edge> edges =  new ArrayList<Edge>();
    public List<Node> roots =  new ArrayList<Node>();
}
