import logging

import azure.functions as func

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sparse

import os
import tempfile
import json
import pickle
from azure.storage.blob import BlobServiceClient, ContainerClient

from . import utils

def load_matrix(matrix, temp_path, file_name):
    file_name = os.path.join(temp_path, file_name)
    with open(file_name, "w+b") as local_file:
        local_file.write(matrix.read())

    ext = file_name[-3:]
    #print("extension name :", ext)
    
    if ext == "npy":
        payload = np.load(local_file.name)
    elif ext == "npz":
        payload = sparse.load_npz(local_file.name)
    elif ext == "pkl":
        with open(local_file.name, "rb") as f:
            payload = pickle.load(f)
    
    return payload


class CollaborativeFilteringRecommender():
    def __init__(self,
                 user_vecs,
                 item_vecs,
                 clicks_sparse,
                 users_arr,
                 items_arr,
                 user_to_sparse_user,
                 sparse_item_to_item):
        
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        self.clicks_sparse = clicks_sparse
        self.users_arr = users_arr
        self.items_arr = items_arr
        self.user_to_sparse_user = user_to_sparse_user
        self.sparse_item_to_item = sparse_item_to_item
        
    
    def rec_items(self, user_id, num_items = 5):
        """
        Calculate and return the recommnded items
        Input :
            - user_id (int) : user for whom recmmendations are calculated
            - num_items (int) : number of recommnded items
        Output :
            - codes (list of int) : recommended items
            - (pd.DataFrame) : recommended items and their attached category (for easier analysis)
        """
        
        user_id = self.user_to_sparse_user[user_id]
        user_ind = np.where(self.users_arr == user_id)[0][0] 
        pref_vec = self.clicks_sparse[user_ind,:].toarray() 
        pref_vec = pref_vec.reshape(-1) + 1 
        pref_vec[pref_vec > 1] = 0 
        rec_vector = self.user_vecs[user_ind,:].dot(self.item_vecs.T) 
       
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
        recommend_vector = pref_vec*rec_vector_scaled 
        
        product_idx = np.argsort(recommend_vector)[::-1][:num_items] 
       
        rec_list = [] 
        for index in product_idx:
            code = self.items_arr[index]
            code = self.sparse_item_to_item[code] 
            rec_list.append(code)
                             
        return rec_list


def main(req: func.HttpRequest,
         matrix1: func.InputStream,
         matrix2: func.InputStream,
         matrix3: func.InputStream,
         matrix4: func.InputStream,
         matrix5: func.InputStream,
         matrix6: func.InputStream,
         matrix7: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get user_id
    user_id = req.params.get("user")
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            user_id = 99 #pass
        else:
            user_id = req_body.get("userId")
    
    user_id = int(user_id)

    
    print("--------------------------------------")
    print("user id = ", user_id)
    print("user_id.type :", type(user_id))

    # Load matrices for prediction
    temp_path = tempfile.gettempdir()
    #print("temp_path :", temp_path)

    user_vecs = load_matrix(matrix1, temp_path, "user_vecs.npy")
    #print("Avec fonction load_matrix : user_vecs.shape =", user_vecs.shape)

    item_vecs = load_matrix(matrix2, temp_path, "item_vecs.npy")
    #print("Avec fonction load_matrix : item_vecs.shape =", item_vecs.shape)

    users_arr = load_matrix(matrix3, temp_path, "users_arr.npy")
    #print("Avec fonction load_matrix : users_arr.shape =", users_arr.shape)

    items_arr = load_matrix(matrix4, temp_path, "items_arr.npy")
    #print("Avec fonction load_matrix : items_arr.shape =", items_arr.shape)

    clicks = load_matrix(matrix5, temp_path, "clicks.npz")
    #print("Avec fonction load_matrix : clicks =\n")
    #print(clicks)

    user_to_sparse_user = load_matrix(matrix6, temp_path, "user_to_sparse_user.pkl")

    sparse_item_to_item = load_matrix(matrix7, temp_path, "sparse_item_to_item.pkl")

    cf_object = CollaborativeFilteringRecommender(user_vecs,
                                                item_vecs,
                                                clicks,
                                                users_arr,
                                                items_arr,
                                                user_to_sparse_user,
                                                sparse_item_to_item)
    
    recommendations = cf_object.rec_items(user_id)

    print("recommendations for user {} :\n{}".format(user_id, recommendations))
    print("type(recommendations :", type(recommendations))
    recommendations_string = [str(i) for i in recommendations]
    recommendations_string = "[" + ", ".join(recommendations_string) + "]"

    recommendations_int = [int(x) for x in recommendations]
    outbound_json = json.dumps(recommendations_int)
    print("outbound_json :", outbound_json)

    if user_id == None:
        return func.HttpResponse(f"Hello {user_id}. This HTTP triggered function executed successfully but could not get any user_id.")
    else:
        return func.HttpResponse(
             outbound_json ,
             status_code=200
        )
