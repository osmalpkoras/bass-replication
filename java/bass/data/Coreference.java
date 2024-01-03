/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;
import java.util.ArrayList;

public class Coreference implements Serializable{

	private static final long serialVersionUID = -2778604144499022942L;
	private ArrayList<Mention> mentions;
	public Boolean visited;
	
	public Coreference() {
		// TODO Auto-generated constructor stub
		mentions = new ArrayList<Mention>();
		this.visited = false;
	}
	
	public Coreference( ArrayList<Mention> mentions) {
		// TODO Auto-generated constructor stub
		this.mentions = mentions;
	}
	
	public void setMentions(ArrayList<Mention> mentions)
	{
		this.mentions = mentions;
	}
	
	public ArrayList<Mention> getMentions()
	{
		return mentions;
	}
	
	public void addMention(Mention mention)
	{
		mentions.add(mention);
	}

}
