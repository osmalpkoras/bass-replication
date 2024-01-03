# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from rouge_score import rouge_scorer
from bert_score import score
import logging
import transformers
import torch
import pickle
import gzip


class Evaluation:
    def __init__(self) -> None:
        self.documents = []
        self.summaries = []
        self.generated_summaries = []
        self.file = None
    
    def load(self, file):
        self.documents = []
        self.summaries = []
        self.generated_summaries = []
        self.file = file
        
        clip_str = lambda x: x.replace("<s>", "").replace("</s>", "").strip()
        with gzip.open(file, "rb") as fin:
            try:
                while True:
                    data = pickle.load(fin)
                    self.documents.extend([clip_str(x) for x in data["documents"]])
                    self.summaries.extend([clip_str(x) for x in data["summaries"]])
                    self.generated_summaries.extend([clip_str(x) for x in data["generated_summaries"]])
            except:
                pass
        pass
        
    def run(self):
        return self.evaluate(self.documents, self.summaries, self.generated_summaries)

    def get_r1_scores(self):
        scores = []
        for rouge in self.rouge_scores:
            scores.append(rouge["rouge1"].fmeasure)

        return scores

    def get_r2_scores(self):
        scores = []
        for rouge in self.rouge_scores:
            scores.append(rouge["rouge2"].fmeasure)

        return scores

    def get_rL_scores(self):
        scores = []
        for rouge in self.rouge_scores:
            scores.append(rouge["rougeL"].fmeasure)

        return scores

    def get_rLsum_scores(self):
        scores = []
        for rouge in self.rouge_scores:
            scores.append(rouge["rougeLsum"].fmeasure)

        return scores

    def get_bert_scores(self):
        return self.bert_scores[2].tolist()


    def evaluate(self, documents, summaries, generated_summaries):
        rouge_scores = []
        with ProcessPoolExecutor() as executor:
            rng = list(range(0, len(summaries)))
            rng = [rng[i:i + 3000] for i in range(0, len(rng), 3000)]
                
            for r in rng:
                futures: set[Future] = set(executor.submit(rouge_scoring, summaries[i], generated_summaries[i]) for i in r)

                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            result = future.result()
                            rouge_scores.append(result)
                        except Exception as ex:
                            pass

            executor.shutdown(wait=False, cancel_futures=False)

        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

        (P, R, F1), hash_code = score(summaries, generated_summaries, lang='en', model_type="roberta-large", rescale_with_baseline=True, return_hash=True, verbose=False)
        
        self.bert_scores = (P, R, F1)
        self.rouge_scores = rouge_scores
        
        Scores = namedtuple("Evaluation", ["R_1", "R_2", "R_L", "R_LSum", "BERTScore"])
        self.scores = Scores(R_1=sum([s["rouge1"].fmeasure for s in rouge_scores]) / len(rouge_scores),
                        R_2=sum([s["rouge2"].fmeasure for s in rouge_scores]) / len(rouge_scores),
                        R_L=sum([s["rougeL"].fmeasure for s in rouge_scores]) / len(rouge_scores),
                        R_LSum=sum([s["rougeLsum"].fmeasure for s in rouge_scores]) / len(rouge_scores),
                        BERTScore=float(torch.mean(F1)))
        return self.scores, len(summaries)
        


def rouge_scoring(summary, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True, split_summaries=True)
    return scorer.score(summary, prediction)
    
