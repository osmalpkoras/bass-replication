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

public class Edge implements Serializable {
    private static final long serialVersionUID = -6830082821555559113L;
    public Node source = null;
    public Node destination = null;
    public String relation = null;
    public boolean is_extra;
    public double weight = 0.;

    @Override
    public String toString() {
        return "("+source.toString()+") "+relation+" -> ("+destination.toString()+")";
    }
}
