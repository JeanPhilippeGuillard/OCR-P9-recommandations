import numpy as np
import pandas as pd
import scipy.sparse as sparse
import os
import pickle
import math

import tempfile
from io import StringIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from credentials import CONNECTION_STRING

import implicit




class CollaborativeFilteringRecommender():
    """
    Recommend top n articles based on collaborative filtering.
    """
    
    def __init__(self, data, items, sample_size=None, save_lookup_dicts=False):
        """
        Input :
            - data (pd.Dataframe) : dataframe containing users / articles interactions
            - items (pd.Dataframe) : dataframe with all articles and their 
                                        their attached information
            - sample_size (int) : number of data lines to consider (to limit calculation time
            if needed)
            - save_lookup_dicts (bool) : if True, calculate lookup dictonaries and save them (long process).
            If False, use already saved dictionaries.
        """

        self.df = data.copy()
        
        if not sample_size:
            self.sample_size = len(self.df)
        else:
            self.sample_size = sample_size
            
        self.items = items
        self.save_lookup_dicts = save_lookup_dicts
        
        self.create_sparse_matrix()


    def inside_check(self):
        # Just used to return internal variables for checking
        return self.user_to_sparse_user
        
    
    def smooth_user_preference(self, x):
        """
        Smooth click counts :
        Input :
            - x (int) : number of clicks on an article for a single user
        Output :
            - (float) : smoothed number of clicks
        """

        return math.log(1+x, 2)
    
    
    def map_ids(self, row, mapper):
        """
        Convert indices between sparse and compressed matrices :
        Input :
            - row (int) : index to convert
            - mapper (dict of int) : dictionary used for conversion
        Output :
            - (int) : converted index
        """

        return mapper[row]

        
    def prepare_data(self):
        """
        Process input dataframe to include only relevant information :
        """
            
        self.df = self.df[["user_id", "click_article_id"]]
        
        self.df.rename(columns={"user_id": "user", "click_article_id": "item"},
                      inplace=True)
        self.df["click_counts"] = 1
        self.df = self.df.groupby(["user", "item"])["click_counts"] \
                    .sum().apply(self.smooth_user_preference).reset_index()
        
        self.df.sort_values(by="user", axis=0, inplace=True)
        # In order to reduce inference time, only consider the fisrt 10.000 users
        # as the mobile app spans from 0 to 9.999
        first_user_to_exclude = self.df[self.df["user"] == 10000].index[0]

        self.df = self.df[:first_user_to_exclude]

        self.df["user"] = self.df["user"].astype("category")
        self.df["item"] = self.df["item"].astype("category")
        
        self.df["user_code"] = self.df["user"].cat.codes
        self.df["item_code"] = self.df["item"].cat.codes
        
        self.item_lookup = self.items[["article_id", "category_id"]].drop_duplicates()
        self.item_lookup.rename(columns={"article_id": "item", "category_id": "category"}, inplace=True)
   
    
    def create_sparse_matrix(self):
        """
        Transform spare data into a scipy sparse matrix to save memory.
        Calculate lookup dictionaries for indices conversions.
        """
                
        self.prepare_data()
        
        item_to_idx = {}
        idx_to_item = {}
        for (idx, item) in enumerate(self.df["item_code"].unique().tolist()):
            item_to_idx[item] = idx
            idx_to_item[idx] = item

        user_to_idx = {}
        idx_to_user = {}
        for (idx, user) in enumerate(self.df["user_code"].unique().tolist()):
            user_to_idx[user] = idx
            idx_to_user[idx] = user
         
        I = self.df["user_code"].apply(self.map_ids, args=[user_to_idx]).values
        J = self.df["item_code"].apply(self.map_ids, args=[item_to_idx]).values
        V = np.ones(I.shape[0])
        
        clicks = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
        self.clicks_sparse = clicks.tocsr()
        
        self.df["user_sparse_code"] = self.df["user_code"].map(user_to_idx)
        self.df["item_sparse_code"] = self.df["item_code"].map(item_to_idx)
        
        self.users_arr = np.sort(self.df["user_sparse_code"].unique()) # Get our unique customers
        self.items_arr = self.df["item_sparse_code"].unique() # Get our unique products that were purchased
        print("users_arr :\n", self.users_arr)
            
        print("calcul des dictionnaires")
        self.item_to_sparse_item = {}
        self.sparse_item_to_item = {}
        for sparse_item in self.df["item_sparse_code"].unique().tolist():
            item = self.df[self.df["item_sparse_code"] == sparse_item]["item"].tolist()[0]
            self.item_to_sparse_item[item] = sparse_item
            self.sparse_item_to_item[sparse_item] = item

        self.user_to_sparse_user = {}
        self.sparse_user_to_user = {}
        for sparse_user in self.df["user_sparse_code"].unique().tolist():
            user = self.df[self.df["user_sparse_code"] == sparse_user]["user"].tolist()[0]
            self.user_to_sparse_user[user] = sparse_user
            self.sparse_user_to_user[sparse_user] = user
                


    def fit_model(self, alpha=15, factors=20, regularization=0.1, iterations=10,
                 pct_test=0.2):
        """
        Calculate users and items matrices that will be used for recommendations (dot product)
        Input :
            - alpha (int) : 
            - factors (int) : second dimension of users and items matrices
            - regularization (float) : regularization coefficient
            - iteration (int) : number od epochs for model training
            - pct_test (float) : percentage of data masked in training set
        Output :
            - user_vecs (np.array) : matrix containing users features
            - item_vecs (np.array) : matrix containing articles features
        """
        
        self.train_set = self.clicks_sparse
        
        self.user_vecs, self.item_vecs = \
            implicit.alternating_least_squares((self.train_set*alpha).astype('double'), 
                                                              factors=factors, 
                                                              regularization=regularization, 
                                                              iterations=iterations)
        
        
        return self.user_vecs, self.item_vecs


    def get_files(self):
        """
        Return training data
        Input : None
        Output :
            - user_vecs (np.array) : matrix containing users features
            - item_vecs (np.array) : matrix containing articles features
            - clicks_sparse (scipy sparse matrix) : users - items interactions matrix
            - user_to_sparse_user (dict of int) : conversion table between regular and sparse users indices
            - sparse_item_to_item (dict of int) : conversion table between sparse and regular items indices
            - user_arr (np.array) : list of users
            - items_arr (np.array) : list of items
        """

        return (self.user_vecs,
                self.item_vecs,
                self.clicks_sparse,
                self.user_to_sparse_user,
                self.sparse_item_to_item,
                self.users_arr,
                self.items_arr)

def load_csv(blob_service_client, container_name, blob):
    """
    Load csv file from an Azure storage container
    Input :
        - blob_service_client (BlobServiceClient) : connection to the Azure storage
        - container_name (str)
        - blob : blob object to load
    Output :
        - df (pd.DataFrame) = dataframe made with csv data
    """

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
    tempo_blob = blob_client.download_blob().content_as_text()
    df = pd.read_csv(StringIO(tempo_blob))

    return df

def save_file(obj, blob_service_client, temp_dir, container_name, file_name):
    """
    Save object as a file to an Azure storage container. The object is first saved on a temporary local directory
    then copied to the container.
    Input :
        - obj : object to save
        - blob_service_client (BlobServiceClient) : connection to the Azure storage
        - temp_dir (str) : name of a temporary directory to save object before copying it to the container
        - container_name (str)
        - file_name (str)
    Output : None
    """

    file_path = os.path.join(temp_dir, file_name)

    if file_name[-3:] == "npy":
        np.save(file_path, obj)
    elif file_name[-3:] == "npz":
        sparse.save_npz(file_path, obj)
    elif file_name[-3:] == "pkl":
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    
    blob_client = blob_service_client.get_blob_client(container_name, file_name)
    with open(file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print("{} saved".format(file_name))


def main():

    # Connect to storage account for loading data
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    print("blob_service_client =", blob_service_client)



    # Load clicks information
    print("Loading data...")
    container_name_for_loading = "input-data"
    container_client_for_loading = blob_service_client.get_container_client(container_name_for_loading)
    df_list = []


    for blob in container_client_for_loading.list_blobs():
        if "hour" in blob.name:
            tempo_file_name = blob.name
            print("Loading ", blob.name)
            df = load_csv(blob_service_client, container_name_for_loading, blob)

            df_list.append(df)

        else:
            print("loading ", blob.name)
            items_df = load_csv(blob_service_client, container_name_for_loading, blob)
        
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Calculate matrices
    print("Calculating model parameters")

    cf_object = CollaborativeFilteringRecommender(df, items_df)
    cf_object.fit_model()
    print("df :")
    print(cf_object.inside_check())

    calculated_parameters = cf_object.get_files()
    calculated_parameters_names = ["user_vecs.npy",
                                "item_vecs.npy",
                                "clicks_sparse.npz",
                                "user_to_sparse_user.pkl",
                                "sparse_item_to_item.pkl",
                                "users_arr.npy",
                                "items_arr.npy"]


    # Save matrices
    temp_dir = tempfile.gettempdir()
    container_name_for_saving = "trained-model"
    #container_client_for_saving = blob_service_client.get_container_client(container_name_for_saving)

    print("Saving model parameters in {}...".format(container_name_for_saving))

    for parameter, file_name in zip(calculated_parameters, calculated_parameters_names):
        save_file(parameter, blob_service_client, temp_dir,container_name_for_saving, file_name)


    print("Model parameters saved")



if __name__ == "__main__":
    main()
