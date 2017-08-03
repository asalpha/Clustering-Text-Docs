import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import csv




# reads the csv and returns a list of list containing all the date
def read_csv(name, pos):
    global data_points
    data_points = 0
    read_file = open(name + '.csv', 'r', encoding="latin1")
    csv_read = csv.reader(read_file)
    lst = []
    for row in csv_read:
        data_points += 1
        print("Extracting Data....")
        lst.append(row[pos])
    read_file.close()
    return lst



# Takes a list of list and write it on a csv row by row
def writecsv(wfile, data):
    for r in data:
        wfile.writerow(r)



# tokenizer function
def wrd_tknizer(sen):
        tokens = word_tokenize(sen)
        lmtzed_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english')]
        return lmtzed_tokens



# Maps everything on to a graph and finds clusters of words
def cluster_sentences(sntcs):

        tfidf_vectorizer = TfidfVectorizer(tokenizer=wrd_tknizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=1.2,
                                        min_df=0,
                                        lowercase=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sntcs).todense()
        print(tfidf_matrix)
        #plt.show()
        print("something")
        max_sc = 0
        global best_cluster
        best_cluster = 0
        a = data_points//50
        b = data_points//3
        for nclust in range(a, b):
            kmeans = KMeans(n_clusters=nclust)
            kmeans.fit(tfidf_matrix)
            label = kmeans.labels_
            sil_coeff = silhouette_score(tfidf_matrix, label, metric='euclidean')
            if sil_coeff > max_sc:
                max_sc = sil_coeff
                best_cluster = nclust
            print("For n_clusters={}, The Silhouette Coefficient is {}".format(nclust, sil_coeff))
        print('Mapping to Graph...')
        pca = PCA(n_components=max_sc).fit(tfidf_matrix)
        data2D = pca.transform(tfidf_matrix)
        plt.scatter(data2D[:,0], data2D[:,1])
        print("Making Clusters...")
        kmeans = KMeans(n_clusters=best_cluster)
        kmeans.fit(tfidf_matrix)
        # centers2D = pca.transform(kmeans.cluster_centers_)
        # plt.hold(True)
        # plt.scatter(centers2D[:,0], centers2D[:,1], 
            # marker='x', s=200, linewidths=3, c='r')
        # plt.show()
        clusters = collections.defaultdict(list)
        print(kmeans)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)



# used to print the clusters
def print_clusters(clust, sent):
    n = 0
    new_data = []
    while n < nclusters:
        title = grams[clusters[n][0]]
        lemmatized_title = lemmatizer.lemmatize(title)
        print ("cluster ",n + 1,":")
        clstr = ''
        val = 0
        switch = 0
        for q,sentence in enumerate(clusters[n]):
            if switch == 0:
                main = grams[sentence]
                switch = 1
            val += int(value[sentence])
            clstr += grams[sentence] + ', '
            print ("\tsentence ",q,": ",grams[sentence])
        # print(val)
        # print(clstr)
        n += 1
        new_data.append([val, main, clstr])
    # print(new_data)
    return new_data



# lemmatizer
lemmatizer = WordNetLemmatizer()


# takes in the path for the file and the column where the 
# information is found to read the value and grams from the csv
value = read_csv('H:\\extra\\5-Gram 44', 1)
grams = read_csv('H:\\extra\\5-Gram 44', 0)
grams = grams[1:]
value = value[1:]


# number of clusters



# main function used to run
clusters = cluster_sentences(grams)

nclusters= best_cluster
# prints the clusters and sentences and assigns the value to be written
# on the csv to new_data
new_data = print_clusters(clusters,grams)


# select the path and file name where you want to write everything
write_file = open('H:\\extra\\5-meh_new.csv', 'w', newline='', encoding="latin1")
csv_write = csv.writer(write_file)
writecsv(csv_write, new_data)
write_file.close()















example = ['Home Folder Please','PWC cust connect frozen', 'PWC Cust connect frozen', 'PWC  Cust connect frozen', 'cust connect frozen',
'Cust connect frozen', 'PEC cust connect frozen', 'Pwc Cust connect frozen', 'PWC cust connect frozen tfsa not loading',
'PWC cust connect frozen error 103','OC - unable to open file on network drive',
'OCSC - missing network drive',
'URGENT OC - unable to access files on network drive',
'OC - Network drive not mapped',
'USOC - windows - add network Drive',
'Folder Please Create',
'MI - Cannot Access Network Drive',
'How to gain access to a network drive.',
'M&I Connect - Map network drive and printer',
'USOC - How To map network drive',
'Map network drive request',
'request to map network drive',
'user needs to be mapped to network drive',
'request to map network drive',
'Please Create Home',
'How to get access to a shared network drive.',
'How to get access to a shared network drive.',
'Create Home Folder'
]