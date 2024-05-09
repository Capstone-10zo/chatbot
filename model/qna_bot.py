from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import ast

# QnABot 클래스
class QnABot:
    def __init__(self, model_path, data_path):
        self.model = SentenceTransformer(model_path)
        self.data = pd.read_csv(data_path)
        # 문자열 -> 파이썬 리스트로 변환
        self.data['embedding'] = self.data['embedding'].map(ast.literal_eval)
    
    def predict_answer(self, text):
        embedding = self.model.encode(text)
        self.data['distance'] = self.data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        output = self.data.loc[self.data['distance'].idxmax()]
        similar_question = output['question']
        predicted_answer = output['answer']
        similarity_score = output['distance']
        return similar_question, predicted_answer, similarity_score
    
# QnABot 클래스를 저장하는 함수
def save_qna_bot(qna_bot, filename):
    with open(filename, 'wb') as f:
        pickle.dump(qna_bot, f)

# 저장된 QnABot 클래스를 불러오는 함수
def load_qna_bot(filename):
    with open(filename, 'rb') as f:
        qna_bot = pickle.load(f)
    return qna_bot