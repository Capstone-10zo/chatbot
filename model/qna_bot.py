from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import ast
import os

# QnABot 클래스
class QnABot:
    def __init__(self, model_path, folder_path):
        self.model = SentenceTransformer(model_path)
        self.data = self.load_data_from_folder(folder_path)
    
    def load_data_from_folder(self, folder_path):
        all_data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                data['embedding'] = data['embedding'].map(ast.literal_eval) # 문자열 -> 파이썬 리스트
                all_data.append(data)
        return pd.concat(all_data, ignore_index=True)
    
    def predict_answer(self, text):
        embedding = self.model.encode(text)
        self.data['distance'] = self.data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        output = self.data.loc[self.data['distance'].idxmax()]
        similar_question = output['question']
        predicted_answer = output['answer']
        similarity_score = output['distance']
        
        if similarity_score < 0.7:
            predicted_answer = '죄송합니다. 질문을 이해하지 못했어요.'

        return similar_question, predicted_answer, similarity_score

    
def save_qna_bot(qna_bot, filename):
    with open(filename, 'wb') as f:
        pickle.dump(qna_bot, f)

def load_qna_bot(filename):
    with open(filename, 'rb') as f:
        qna_bot = pickle.load(f)
    return qna_bot