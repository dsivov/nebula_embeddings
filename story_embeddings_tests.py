from sentence_transformers import SentenceTransformer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from arango import ArangoClient
from simpleneighbors import SimpleNeighbors
from nebula_model import NEBULA_DOC_MODEL
from gensim.models.doc2vec import TaggedDocument


def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

def get_stories_from_db(db):
    query = 'FOR doc IN Stories RETURN doc'
   
    stories = {}
    cursor = db.aql.execute(
            query
        )
    for data in cursor:
        #print(data)
        stories[data['movie_id']]=data
    return(stories)

def create_bert_embeddings(stories):
    sentences = []
   
    embedding_dimensions = 768
    single_index = SimpleNeighbors(embedding_dimensions)
    for story in stories.values():
        sentence = TreebankWordDetokenizer().detokenize(story['story'][0])
        #print(sentence)
        sentences.append(sentence)
    
    sbert_model = SentenceTransformer('stsb-roberta-base')
    sentence_embeddings = sbert_model.encode(sentences, show_progress_bar=True)
    for embedding, key in zip(sentence_embeddings, stories.values()):
        single_index.add_one(key['movie_id'], embedding)
    

        # print("Key: ", key['movie_id'])
        # print("Embedding: ", embedding)
        # print("Vector Len: ", len(embedding))
        # input("Press any key...")
    return(single_index)

def create_doc2vec_embeddings(stories):
    sentences = []
    tags = {}
    embedding_dimensions = 160
    single_index = SimpleNeighbors(embedding_dimensions)
    for i, story in enumerate(stories.values()):  
        dfs_doc = TaggedDocument(words=story['story'][0], tags=[story['story'][1][0]]) 
        sentences.append(dfs_doc)
        tags[i] = story['story'][1][0]
        # print(story['movie_id'])
        # print(dfs_doc)
        # input("Press....")
        
    model = NEBULA_DOC_MODEL(dimensions=embedding_dimensions, epochs=400)
    model.fit(sentences, tags)
    print(len(sentences), " ", len(tags))
    sentence_embeddings = model._get_embeddings(tags)
    for embedding, key in zip(sentence_embeddings, stories.values()):
        single_index.add_one(key['movie_id'], embedding)
    return(single_index)

def main():
    num_index_trees = 512
    db = connect_db("nebula_dev")
    stories = get_stories_from_db(db)
    index = create_doc2vec_embeddings(stories)
    #index = create_bert_embeddings(stories)
    index.build(n=num_index_trees) 
    index.save("nebula_index_single")
    while (True):
        _key = input("Enter movie id: ")
        sims = index.neighbors(_key, n=10)
        print("----------------------") 
        print("Top 10 Sims for Movie: " + _key)
        for sim in sims:
            print(sim)
            #print(sim)
        print("----------------------")  


if __name__ == '__main__':
    main()