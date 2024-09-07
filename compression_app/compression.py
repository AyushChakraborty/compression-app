import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

#tasks to be done 
        #till needs to be improved since it just saves it
        #locally on the same folder 

        #take files till max limit of 500kB

        #provide options: soft compression, casual compression, hard compression, where we can just
        #change the value of k
        
        #have a section explaining the concept behind k means compression 

        #take care of the case when file more than the limit is uploaded 
        

#find the centroid index value closest to any data point and store it in a list, this is the step of forming
#the clusters
def find_closest_centroids(X, centroids):

    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray (m, n)): Input values      
        centroids (ndarray (K, n)): centroids matrix which stores the centroid vector in its rows
    
    Returns:
        idx (ndarray (m,)): closest centroids to xi
    """
    
    idx = []
    
    for i in range(X.shape[0]):
        diff = X[i]-centroids
        diff = diff**2
        dist = np.sum(diff, axis=1) #along cols
        minimum_ele = np.min(dist)
        idx.append(np.where(dist == minimum_ele)[0][0])
        
    idx =  np.array(idx)
    
    return idx

#func to reassign centroids of a cluster to the mean of all the data pts of that cluster
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray (m, n)): Data points
        idx (ndarray (m,)): Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int): number of centroids
    
    Returns:
        centroids (ndarray (K, n)): New centroids computed
    """
    
    centroids = []
    
    for i in range(K):
        cluster_pts = X[np.where(idx == i)]
        cluster_avg = np.sum(cluster_pts, axis=0)/(cluster_pts.shape[0])
        centroids.append(cluster_avg)
    centroids = np.array(centroids)
            
    return centroids

#the main func where k-means is run
def run_kmeans(X, init_centroid, max_iter):
    st.write("Training in progress.....")
    centroid = init_centroid
    K = init_centroid.shape[0]
    for i in range(max_iter):
        idx = find_closest_centroids(X, centroid)
        centroid = compute_centroids(X, idx, K)
        st.write(f"iteration {i+1} done")

    return idx, centroid

#a good practice is to initially assign some data points themselves as the centroids, so a func for this
def init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray (m,n)): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray (K,n)): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def reduce_rgb(X):
    """
    This function is to be called when the image fed is not png

    Args:
        X (ndarray (m,n)): Data points
    Returns:
        X_scaled (ndarray(m,n)): Scaled rgb values between 0 and 1
    """

    X_scaled = X/255

    return X_scaled


st.write("Compress your images!!")

file = st.file_uploader("Choose a file to compress", type=["jpg", "jpeg", "png"], )

open = False

if file is not None:
    st.write("Image accepted!")
    image = Image.open(file)
    img_np = np.array(image)
    st.image(image, caption="input image without compression")
    print(img_np.shape)
    open = True
else: 
    print("Please upload an image")

if open: 
    X = np.reshape(img_np, (img_np.shape[0]*img_np.shape[1], 3))
    if file.type != "png":
        X = reduce_rgb(X)
    
    k = 20   #later how much compression option can be added 20
    max_iter = 10
    init_centoids = init_centroids(X, k)
    idx, centroids_final = run_kmeans(X, init_centoids, max_iter)

    X_final = centroids_final[idx, :]
    X_final = np.reshape(X_final, img_np.shape)

    image_int = (X_final*255).astype(np.uint8)
    image_compressed = Image.fromarray(image_int)
    st.image(image_compressed)   #displaying the final compressed image

    #downloading the image
    if st.button("Download the compressed image"):
        image_compressed.save("Compressed_img.jpeg", "JPEG") 

