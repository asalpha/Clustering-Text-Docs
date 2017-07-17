import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import csv


#######################################################################################

def read_csv(name, pos):
    read_file = open(name + '.csv', 'r', encoding="latin1")
    csv_read = csv.reader(read_file)
    lst = []
    for row in csv_read:
        print("Extracting Data....")
        lst.append(row[pos])
    read_file.close()
    return lst



def writecsv(wfile, data):
    for r in data:
        print('Almost Done....')
        wfile.writerow(r)
        print('----------------')




#######################################################################################
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

# tokenizer function
def wrd_tknizer(sen):
        tokens = word_tokenize(sen)
        print('1111111111111111111')
        print(tokens)
        print('1111111111111111111')
        lmtzed_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english')]
        print('22222222222222222222')
        print(lmtzed_tokens)
        print('22222222222222222222')
        return lmtzed_tokens


def cluster_sentences(sntcs, no_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=wrd_tknizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.9,
                                        min_df=0,
                                        lowercase=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sntcs)
        kmeans = KMeans(n_clusters=no_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)


def print_clusters(clust, sent):
    n = 0
    new_data = []
    while n < nclusters:
        title = example1[clusters[n][0]]
        lemmatized_title = lemmatizer.lemmatize(title)
        #print ("cluster ",n + 1,":", lemmatized_title)
        print ("cluster ",n + 1,":")
        clstr = ''
        val = 0
        for q,sentence in enumerate(clusters[n]):
            main = example1[sentence]
            val += int(value[sentence])
            clstr += example1[sentence] + ', '
            print ("\tsentence ",q,": ",example1[sentence])
        print(val)
        print(clstr)
        n += 1
        new_data.append([val, main, clstr])
    print(new_data)
    return new_data
example1 = read_csv('H:\\extra\\blah', 0)
value = read_csv('H:\\extra\\blah', 1)
# lemmatizer
lemmatizer = WordNetLemmatizer()
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
# number of clusters
nclusters= 180

# main function used to run
clusters = cluster_sentences(example1, nclusters)
print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
# prints the clusters and sentences
x = print_clusters(clusters,example1)
#print(clusters)

write_file = open('H:\\extra\\meh_new.csv', 'w', newline='', encoding="latin1")
csv_write = csv.writer(write_file)

writecsv(csv_write, x)
write_file.close()
print('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC')
