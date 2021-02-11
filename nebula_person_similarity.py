from arango import ArangoClient
from numpy.core.numeric import correlate
from scipy.spatial import distance
from scipy.stats import entropy
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim import similarities

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

def get_embeddings_from_db(db, algo):
    query = 'FOR doc IN Embedding FILTER doc.algo == @algo RETURN doc'
    bind_vars = {'algo': algo}
    embeddings = {}
    cursor = db.aql.execute(
            query,
            bind_vars=bind_vars
        )
    for data in cursor:
        embeddings[data['actor_id']]=data
    return(embeddings)

def get_requested_movie(db, movie_id, algo):
    query = 'FOR doc IN Embedding FILTER doc.movie_id == @movie_id AND doc.algo == @algo RETURN doc'
    bind_vars = {'movie_id': movie_id, 'algo': algo}
    embeddings = {}
    cursor = db.aql.execute(
            query,
            bind_vars=bind_vars
        )
    for data in cursor:
        embeddings[data['movie_id']]=data
    return(list(embeddings.values()))

def nebula_check_distance(embeddings, f_vec, algo):
    count = 0    
    correlations = []
    model = Doc2Vec.load("model.dat")
    similarities.Similarity
    # test_text = ['person1', 'carry/hold (an object)', 'stand', 'sit', 'wearing', 'glass', 'holding', 'cup', 'holding', 'phone', 'wearing', 'jacket', 'wearing', 'tie', 'With()', 'glasses',  
    # 'tableware', 'Then', 'person2', 'carry/hold (an object)', 
    # 'sit', 'stand', 'walk', 'talk to (e.g., self, a person, a group)', 'wearing', 'jacket', 'wearing', 'glass', 'wearing', 'tie', 
    # 'wearing', 'shoe', 'wearing', 'glove', 'holding', 'phone', 'With', 'glasses', 'With', 'tie', 'With()', 'outerwear']
    test_text =  f_vec[0]['story'][0]
    #print(test_text)
    #s_vec = model.infer_vector(test_text)
    for _vec in embeddings.values():
        movie_int_id = _vec['movie_id'].split("/")
        #print(int(movie_int_id[1]))
        cor = distance.correlation(_vec['embeddings'], f_vec[0]['embeddings'])
        cos = distance.cityblock(_vec['embeddings'], f_vec[0]['embeddings'])
        #cos1 = distance.correlation (_vec['embeddings'],s_vec)

        correlations.append([_vec['actor_id'], _vec['actor_name'], 
        _vec['movie_id'], _vec['movie_name'],cor, cos])
        count = count + 1
    sorted_correlations = sorted(correlations, key=lambda corr: corr[4])
    sorted_cosine = sorted(correlations, key=lambda corr: corr[5])
    #sorted_anti_correlations = sorted(correlations, key=lambda corr: corr[4],reverse= True)
    #sorted_s_correlations = sorted(correlations, key=lambda corr: corr[5])
    # print ("Top 10 "+ algo + " Positive Correlations for: ",  f_vec[0]['actor_name'], " ",
    #      f_vec[0]['actor_id'],
    #     " From Movie: ",  f_vec[0]['movie_id'],"(",  f_vec[0]['movie_name'],")")
    print("----------------------") 
    print("Top 10 "+ algo + " Positive Correlations for Movie:" + f_vec[0]['movie_id'],":",f_vec[0]['movie_name'])
    print("----------------------") 
    # print("Story is:")
    # print(test_text)  
    for i, corrs in enumerate(sorted_correlations):
        #print(corrs)
        if (corrs[2] != f_vec[0]['movie_id']):
            print(corrs[4], " ",  corrs[2]," ", corrs[3])
        else:
            i = i - 1 
        if i == 10:
            break
    print("----------------------") 
    print("Top 10 "+ algo + " Positive Cosines for Movie:" + f_vec[0]['movie_id'],":",f_vec[0]['movie_name'])
    print("----------------------") 
    # print("Story is:")
    # print(test_text)  
    for i, corrs in enumerate(sorted_cosine):
        #print(corrs)
        if (corrs[2] != f_vec[0]['movie_id']):
            print(corrs[5], " ",  corrs[2]," ", corrs[3])
        else:
            i = i - 1 
        if i == 10:
            break
    # print("Top 10 "+ algo + " Predicted Correlations")
    # print("----------------------")
    # for i, corrs in enumerate(sorted_s_correlations):
    #     if (corrs[2] != f_vec[0]['movie_id']):
    #         print(corrs[5], " ",  corrs[2]," ", corrs[3])
    #     else:
    #         i = i - 1 
    #     if i == 10:
    #         break
    # print ("Top 10 "+ algo + " Negative Correlations")
    # print("----------------------")
    # for i, corrs in enumerate(sorted_anti_correlations):
    #     if (corrs[2] != f_vec[0]['movie_id']):
    #         print(corrs[4], " ",  corrs[2]," ", corrs[3])
    #     else:
    #         i = i - 1 
    #     if i == 10:
    #         break
    # print("----------------------")
    
    print("=============================DONE with " + algo + "============================")
    print()
    print()
    


def main():
    if len(sys.argv) < 3:
        print("Usage: ", sys.argv[0], " db_name, actor_id")
        exit()
    db_name = sys.argv[1]
    movie_id = sys.argv[2]
    db = connect_db(db_name)
    for algo in ["NEBULA_DOC"]:
        f_vec = get_requested_movie(db, movie_id, algo)
        embeddings = get_embeddings_from_db(db, algo)  
        nebula_check_distance(embeddings,  f_vec, algo)
    for algo in ["NEBULA_WORD"]:
        f_vec = get_requested_movie(db, movie_id, algo)
        embeddings = get_embeddings_from_db(db, algo)  
        nebula_check_distance(embeddings,  f_vec, algo)
if __name__ == '__main__':
    main()