
import settings
from google.cloud import storage as gcs
from datetime import datetime
import re
import json
from annoy import AnnoyIndex
import tempfile
import os

class AnnoyIndexModel:
    def __init__(self):
        client = gcs.Client(settings.PROJECT_ID)
        self.bucket = client.get_bucket(settings.BUCKET_NAME)
        self.n_dim = settings.N_DIM
        
    def build_index(self, user_vec_dic):
        annoy_index = AnnoyIndex(self.n_dim, metric='euclidean')
        index = 0
        user_id_dic = {}

        for user_id, vec in user_vec_dic.items():
            annoy_index.add_item(index, vec)
            user_id_dic[index] = user_id
            index += 1
            
        annoy_index.build(n_trees=10)
        
        _, temp_local_file = tempfile.mkstemp(suffix=".ann")
        
        annoy_index.save(temp_local_file)
        now_datetime_str = re.sub('[-:\.\s]', '', str(datetime.now()))
        filename = 'annoy_model_{}.ann'.format(now_datetime_str)
        blob = self.bucket.blob('{}/{}/{}'.format(settings.BUCKET_NAME, settings.MODEL_SAVE_DIR, filename))
        blob.upload_from_filename(temp_local_file)
        os.remove(temp_local_file)
        
        index_dic = json.dumps(user_id_dic)
        filename = 'index_dic_{}.json'.format(now_datetime_str)
        blob = self.bucket.blob('{}/{}/{}'.format(settings.BUCKET_NAME, settings.INDEX_DIC_SAVE_DIR, filename))
        blob.upload_from_string(index_dic)
        
    def load_gcs_files(self):
        blobs = [b for b in self.bucket.list_blobs(prefix="{}/{}/".format(settings.BUCKET_NAME, settings.INDEX_DIC_SAVE_DIR))]
        blob = self.bucket.blob(blobs[-1].name)
        index_dic = {int(k): v for k, v in json.loads(blob.download_as_string().decode()).items()}
        
        blobs = [b for b in self.bucket.list_blobs(prefix="{}/{}/".format(settings.BUCKET_NAME, settings.MODEL_SAVE_DIR))]
        blob = self.bucket.blob(blobs[-1].name)
        _, temp_local_file = tempfile.mkstemp(suffix=".ann")
        blob.download_to_filename(temp_local_file)
        annoy_index = AnnoyIndex(self.n_dim, metric='euclidean')
        annoy_index.load(temp_local_file)
        #一時ファイルを削除する。
        os.remove(temp_local_file)
        
        return annoy_index, index_dic
        
        
        
        