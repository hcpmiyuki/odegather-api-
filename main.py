from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import traceback
from utils.calc_user_vec import FireStoreOperator
from utils.annoy_index import AnnoyIndexModel

from logging import getLogger

import pickle

# fast api設定
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = getLogger(__name__)

'''
流れ
- レビューの収集して前処理して、ベクトル計算してfirestoreに入れる
- ユーザーベクトル計算する
- インデックスをビルドする
- 推論のエンドポイント作る
'''

@app.get('/build-index', status_code=200)
async def build_index():
    try:
        # ユーザーベクトルの用意
        firestore_utils = FireStoreOperator()
        firestore_utils.calculate_and_set_place_vectors()
        firestore_utils.calculate_and_set_user_vectors()
        
        # インデックスをビルドする
        annoy_index = AnnoyIndexModel()
        annoy_index.build_index(firestore_utils.user_vec_dic)
        
        return {'msg': 'success!'}
    except Exception as e:
        t = traceback.format_exception_only(type(e), e)
        logger.error(t)
        raise HTTPException(status_code=500, detail=t)
    
        return {'error': t}
    
@app.get('/reccomend-users', status_code=200)
async def build_index(user_id:str, recommend_user_count:int):
    try:
        annoy_index_model = AnnoyIndexModel()
        model, index_to_user_id = annoy_index_model.load_gcs_files()
        user_id_to_index_id = {v:k for k, v in index_to_user_id.items()}

        if user_id in user_id_to_index_id:
            user_index = user_id_to_index_id[user_id]
            recommended_user_indexs = model.get_nns_by_item(user_index, recommend_user_count, search_k=-1, include_distances=False)
            recommended_user_ids = list(map(lambda x: index_to_user_id[x], recommended_user_indexs))
            
            return {'results': recommended_user_ids}
        else:
            return {'error': 'user not included in annoy index.'}
        
    except Exception as e:
        t = traceback.format_exception_only(type(e), e)
        logger.error(t)
        raise HTTPException(status_code=500, detail=t)
    
        return {'error': t}

    