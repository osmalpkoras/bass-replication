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

public class Token  implements Serializable {
    private static final long serialVersionUID = 3240144004803444313L;
    public String text = null;
    public String xpos = null;
    public Node node = null;
    public Token dependency_head = null;
    public List<Token> dependency_children =  new ArrayList<Token>();
    public String dependency_relation = null;
    public int index_in_sentence = -1;
    public int index_in_document = -1;
    public int sentence_index = -1;
    public Sentence sentence = null;
    public int copy_count = -1;
    public int begin_pos = -1;
    public int end_pos = -1;
    public Object original = null;

    @Override
    public String toString() {
        return text;
    }
}
