"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
import string
#import difflib
from scipy import ndimage


def reduce_dimensions(feature_vectors_full, model):
    """Reduce dimensions upto 20 principla componenets.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    #get eigenvector calculated from train_data
    v=np.array(model['eigenvector'])

    # Centre the data and transforming it upto 20 dimensions
    pca_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)

    return pca_data


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)

    width = max(image.shape[1] for image in images)

    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    
    filtered_images=[]
    #by using median filter, moderate the noise level
    for i in range (len(images)):
        filtered_images.append(ndimage.median_filter(images[i], 3))
    
    fvectors = np.empty((len(images), nfeatures))

    for i, image in enumerate(filtered_images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)

        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    #get the eigenvector to get 20 principal components
    covx = np.cov(fvectors_train_full, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 20, N - 1))
    v = np.fliplr(v)

    #put this eigenvector into the dictionary to use it again for test_data
    model_data['eigenvector'] = v.tolist()

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
    
    #Tried 40 principal components but it seems not better
    '''
    d12=np.zeros(40)
    indices = 9, 25
    lowercase_list = list(string.ascii_lowercase)
    valid_characters = [i for j, i in enumerate(lowercase_list) if j not in indices]
    #extralist = ['l','’',',','.']
    #finlist =valid_characters+extralist
    for char1 in valid_characters:
        char1_data = fvectors_train[labels_train==char1, :]
        for char2 in valid_characters:
            char2_data = fvectors_train[labels_train==char2, :]
            d12 += divergence(char1_data, char2_data)

    sorted_indexes = np.argsort(-d12)
    features = sorted_indexes[0:10]
    model_data['features'] = features.tolist()

    fvector_train_final = fvectors_train[:, features]
    model_data['fvectors_train'] = fvector_train_final.tolist() for 40 principal
    '''
    
    model_data['fvectors_train'] = fvectors_train.tolist()
    
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)

    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    
    fvectors_train = np.array(model['fvectors_train'])
    train_label = np.array(model['labels_train'])
    
    #I did the divergence step again to get the same feature columsn that i used for train data
    d12=np.zeros(20) #creating empty space for adding divergence below
    indices = 9, 25 #j and z that are not helpful for the divergence
    lowercase_list = list(string.ascii_lowercase)
    #remove j and z in the list
    valid_characters = [i for j, i in enumerate(lowercase_list) if j not in indices]
    #Tried to add some symbols for the divergence but no improvement
    #extralist = ['l','’',',','.']
    #finlist =valid_characters+extralist
    
    for char1 in valid_characters:
        char1_data = fvectors_train[train_label==char1, :]
        for char2 in valid_characters:
            char2_data = fvectors_train[train_label==char2, :]
            d12 += divergence(char1_data, char2_data)
    
    #Find the 10 best features with the divergence calculated above
    sorted_indexes = np.argsort(-d12)
    features = sorted_indexes[0:10]
    
    #should return 10 columns always
    return fvectors_test_reduced[:, features]


def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12


def classify_page(page, model):
    """Perform nearest neighbour classification.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    fvectors_train = np.array(model['fvectors_train'])
    train_label = np.array(model['labels_train'])
    
    #adata = fvectors_train[train_label == '.'] -- it is for finding low frenquency of characters
    #print(adata.shape)
    d12=np.zeros(20) #creating empty space for adding divergence below
    #exclude if there are not many sample character like j and z
    indices = 9, 25 #j and z
    lowercase_list = list(string.ascii_lowercase)
    valid_characters = [i for j, i in enumerate(lowercase_list) if j not in indices]
    #extralist = ['l','’',',','.']
    #finlist =somelist+extralist
    #Get the divergence between characters and then find the best 10 features
    for char1 in valid_characters:
        char1_data = fvectors_train[train_label==char1, :]
        for char2 in valid_characters:
            char2_data = fvectors_train[train_label==char2, :]
            d12 += divergence(char1_data, char2_data)

    sorted_indexes = np.argsort(-d12)
    features = sorted_indexes[0:10]
    
    train = fvectors_train[:, features]
    #test has already been selected with 10 best features
    test = page
    
    #Super compact implementation of nearest neighbour
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1)
    output_labels = train_label[nearest]
    print("done")
    
    return output_labels

def correct_errors( page, labels, bboxes, model):
    """Get the word from output labels and then compare this word with wordlist and then if the word is not in the woord list
       then get the similar words and then change the original word to one of them(first thing found).
    parameters: 
    page - 2d array, each row is a feature vector to be classified 
    labels - the output classification label for each feature vector 
    bboxes - 2d array, each row gives the 4 bounding box coords of the character 
    model - dictionary, stores the output of the training stage 
    """
    '''
    word=""
    wordlist=[]
    index=0
    while index<len(bboxes)-1500: #it takes too much time if all of the characters are tested
        for i in range (index, len(bboxes)-1500):
            # if the character has space more than 4, it is the start of the word ( The others are symbols)
            if bboxes[i,2]-bboxes[i,0]>4:
                for j in range(i+1, len(bboxes)):
                    #if the space is betwwen the character and the next character bigger than 6, that is the end of the word
                    #if it is less than minus 100 that means new line starts, so also end of the word
                    if (bboxes[j,0]-bboxes[j-1,2])>6 or (bboxes[j,0]-bboxes[j-1,2])<(-100) :
                        #from i to j, accumlate the letters to build up a word
                        for k in range (i, j):
                            word= word + labels[k]
                        # if the word has symbols at the front or at the end, then remove it.
                        if len(word)>1:
                            if word[len(word)-1] == "," or word[len(word)-1] == "—" or word[len(word)-1] == "." or word[len(word)-1] == "?" or word[len(word)-1] == "!" or word[len(word)-1] == ";" or word[len(word)-1] == "’" or word[len(word)-1] == "|" or word[len(word)-1] == ":":
                                word = word.replace(word[len(word)-1],"")
                            if word[0] == "," or word[0] == "—" or word[0] == "." or word[0] == "?" or word[0] == "!" or word[0] == "’" :
                                word = word.replace(word[0],"")
                            print("The word has beenn found")
                            print(word)
                            #capital letters are not in the wordlist, so excluded.
                            if not word[0].isupper():
                                wordlist= open("wordlist.txt", 'r').read().split()
                                if not word in (w for i, w in enumerate(wordlist) if i != 1): #to check word is in the wordlist or not
                                    print("This word is not in the worldlist")
                                    #get the similar words from the wordlist
                                    similarwords=difflib.get_close_matches(word, wordlist)
                                    print(similarwords)
                                    for a in range (len(similarwords)): #to find which characters are different between two words
                                        if len(similarwords[a]) == len(word):
                                            print("We are founding this word")
                                            print(word)
                                            print(similarwords[a])
                                            d=[i for i in range(len(word)) if similarwords[a][i] != word[i]] #to find different character index
                                            print("following idexes are different")
                                            print(d)
                                            #change the original word into the word found from the wordlist
                                            for p in d:
                                                labels[i+p]= similarwords[a][p]
                                            break #if one same lenght similar word is found, no need to iterate
                        word="" #clean the word to empty string to get next word
                        index = i+(j-i) # start the loop again with the index of next word not the next letter.
                        break
            else:
                index+=1 #if the first letter found is a symbol then search for the next letter
            break
        '''    
    return labels
