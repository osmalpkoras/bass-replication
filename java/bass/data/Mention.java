/**
 * This source code was kindly provided as is by Wu et al. (BASS: Boosting Abstractive Summarization with Unified Semantic Graph, ACL-IJCNLP 2021)
 * @author Wu et al.
 */
package bass.data;

import java.io.Serializable;

public class Mention implements Serializable{

	private static final long serialVersionUID = -7414824496822993411L;
	private Boolean representative;
	private Boolean visited;
	private Integer sentence_idx;
	private Integer word_start_idx;
	private Integer word_end_idx;
	private Integer head_idx;
	
	public Mention(){
		
	}
	
	public Mention(Boolean representative,Integer sentence_idx,Integer word_start_idx,Integer word_end_idx,Integer head_idx){
		this.representative = representative;
		this.sentence_idx = sentence_idx;
		this.word_start_idx = word_start_idx;
		this.word_end_idx = word_end_idx;
		this.head_idx = head_idx;
		this.visited = false;
	}
	
	public void setRepresentative(Boolean representative)
	{
		this.representative = representative;
	}
	public Boolean getRepresentative()
	{
		return representative;
	}
	public Boolean getVisited()
	{
		return visited;
	}
	public void setVisited()
	{
		this.visited = true;
	}
	
	
	public void setSentence_idx(Integer sentence_idx)
	{
		this.sentence_idx = sentence_idx;
	}
	public int getSentence_idx()
	{
		return sentence_idx;
	}
	
	public void setWord_start_idx(Integer word_start_idx)
	{
		this.word_start_idx = word_start_idx;
	}
	public int getWord_start_idx()
	{
		return word_start_idx;
	}
	
	public void setWord_end_idx(Integer word_end_idx)
	{
		this.word_end_idx = word_end_idx;
	}
	public int getWord_end_idx()
	{
		return word_end_idx;
	}
	
	public void setHead_idx(Integer head_idx)
	{
		this.head_idx = head_idx;
	}
	public int getHead_idx()
	{
		return head_idx;
	}
	
	@Override
	public int hashCode() {
		int result = 0;
		result += 29*this.sentence_idx.hashCode();
		result += 29*this.word_start_idx.hashCode();
		result += 29*this.word_end_idx.hashCode();
		result += 29*this.head_idx.hashCode();
		return result;
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == null) {
			return false;
		}

		if (!(o instanceof Mention)) {
			return false;
		}

		Mention otherNode = (Mention) o;
		int sentence_id = otherNode.getSentence_idx();
		int word_start_id = otherNode.getWord_start_idx();
		int word_end_id = otherNode.getWord_end_idx();
		int head_id = otherNode.getHead_idx();
		
		if(this.sentence_idx==sentence_id&&this.word_start_idx==word_start_id
				&&this.word_end_idx==word_end_id&&this.head_idx==head_id)
			return true;
		
		return false;
	}
}
