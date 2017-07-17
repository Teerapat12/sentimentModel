import pandas as pd
import numpy as np
dfToken = pd.read_csv("../dataset/tokenized_comment_all_2.tsv", error_bad_lines=False,encoding='utf-8',sep='\t')
dfScore = pd.read_csv("../dataset/facebook_comment_annotated_170717_all.tsv", error_bad_lines=False,encoding='utf-8',sep='\t')
dfScore = dfScore[pd.notnull(dfScore['message'])]
dfScore.reset_index(inplace=True)
tokenizedCol = dfToken['tokenized']
dfScore['token'] = tokenizedCol
dfScore['service'][dfScore['service']==True] = 'TRUE'
dfScore.to_csv('../dataset/facebook_comment_tokenized_scored.tsv',sep='\t',encoding='utf-8')