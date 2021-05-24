import re
import emoji
import mojimoji
import neologdn

from janome.tokenizer import Tokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import numpy as np

import settings


    

class Cleansing:
    def __call__(self, text):
        return self.cleansing_text(text)

    # replace and\s to space
    def cleansing_space(self, text):
        return re.sub("\u3000|\s", " ", text)

    # remove URLs
    def cleansing_url(self, text):
        return re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" , text)

    # remove pictographs
    def cleansing_emoji(self, text):
        return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)

    # replace number to zero
    def cleansing_num(self, text):
        text = re.sub(r'\d+', "0", text)
        return text

    # unify characters
    def cleansing_unity(self, text):
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        return text

    def cleansing_text(self, text):
        text = self.cleansing_space(text)
        text = self.cleansing_url(text)
        text = self.cleansing_emoji(text)
        text = self.cleansing_unity(text)
        text = self.cleansing_num(text)
        text = neologdn.normalize(text)
        return text


class JanomeTokenizer:
    def __init__(self):
        self.t = Tokenizer()
        
    def __call__(self, reviews):
        return self.tokenizer(reviews)

    def tokenizer(self, reviews):
        wakati = ''
        for review_text in reviews:
            for token in self.t.tokenize(review_text):  # 形態素解析
                hinshi = (token.part_of_speech).split(',')[0]  # 品詞情報を取得
                hinshi_2 = (token.part_of_speech).split(',')[1]  # 品詞情報の2項目目を取得
                if hinshi in ['名詞']:  # 品詞が名詞、動詞、形容詞の場合のみ以下実行
                    if not hinshi_2 in ['空白','*']:  # 品詞情報の2項目目が空白か*の場合は以下実行しない
                        word = str(token).split()[0]  # 単語を取得
                        if not ',*,' in word:  # 単語に*が含まれない場合は以下実行
                            wakati = wakati + word +' ' # オブジェクトwakatiに単語とスペースを追加 

        return wakati


class FireStoreOperator:
    def __init__(self):
        self.db = self.set_db()
        self.place_vec_dic = {}
        self.user_place_dic = {}
        self.user_vec_dic = {}
        
    def set_db(self):
        # 初期化済みかを判定する
        if not firebase_admin._apps:
            # 初期済みでない場合は初期化処理を行う
            firebase_admin.initialize_app()
        
        return firestore.client()
    
    def get_wakati_reviews(self):
        cleaner = Cleansing()
        tokenizer = JanomeTokenizer()

        all_places_docs = self.db.collection('all_places').get()
        review_data = []

        for doc in all_places_docs:
            place_data = doc.to_dict()

            if 'reviews' in place_data and 'wakati_reviews' not in place_data:
                reviews = place_data['reviews']
                _reviews = list(map(cleaner, reviews))
                wakati_reviews = tokenizer(_reviews)
                doc.reference.update({'wakati_reviews': wakati_reviews})
            else:
                wakati_reviews = place_data['wakati_reviews']
            
            review_data.append({'place_id': doc.id, 'wakati_reviews': wakati_reviews})
        
        self.df = pd.DataFrame(review_data)
        
    def calculate_review_tfidf_vectors(self):
        wakati_list_np = np.array(self.df['wakati_reviews'].values.tolist())
        # vectorizerの生成。token_pattern=u'(?u)\\b\\w+\\b'で1文字の語を含む設定
        vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b', max_features=settings.N_DIM)
        # transformerの生成。TF-IDFを使用
        transformer = TfidfTransformer()

        tf = vectorizer.fit_transform(wakati_list_np) # ベクトル化
        tfidf = transformer.fit_transform(tf) # TF-IDF
        
        self.place_vec_dic = {place_id: vector for place_id, vector in zip(self.df['place_id'].values.tolist(), list(tfidf.toarray())) if sum(vector) != 0.}
    
    def set_vectors(self, vector_dic, collection_name):
        for _id, vector in vector_dic.items():
            doc_ref = self.db.collection(collection_name).document(_id)
            doc_ref.update({'vector': list(vector)})
            
    def calculate_and_set_place_vectors(self):
        self.get_wakati_reviews()
        self.calculate_review_tfidf_vectors()
        self.set_vectors(self.place_vec_dic, 'all_places')
        
    def get_user_places(self):
        docs = self.db.collection('users').get()

        for doc in docs:
            self.user_place_dic[doc.id] = []
            
            list_docs = doc.reference.collection('lists').get()
            for list_doc in list_docs:
                place_docs = list_doc.reference.collection('places').get()
                for place_doc in place_docs:
                    self.user_place_dic[doc.id].append(place_doc.id)
                    
    def calculate_user_vectors(self):
        for user_id, place_ids in self.user_place_dic.items():
            vectors = []
            for place_id in place_ids:
                if place_id in self.place_vec_dic:
                    vectors.append(self.place_vec_dic[place_id])
            
            if vectors != []:
                user_vec = np.mean(vectors, axis=0)
            else:
                user_vec = [0.] * 1000
                
            self.user_vec_dic[user_id] = user_vec
            
    def calculate_and_set_user_vectors(self):
        self.get_user_places()
        self.calculate_user_vectors()
        self.set_vectors(self.user_vec_dic, 'users')
        

    
        

            
            