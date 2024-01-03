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

import java.io.Serializable;

public class SummarizationDatapoint extends Datapoint  implements Serializable {
    public String document_text;
    public String summary_text;
}

