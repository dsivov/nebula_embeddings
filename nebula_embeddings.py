from sys import prefix
from gensim.models.doc2vec import TaggedDocument

from multiprocessing.managers import all_methods
import networkx as nx
import numpy as np
from scipy.ndimage.measurements import label
from scipy.stats.stats import mode
from nebula_networkx_adapter import Nebula_Networkx_Adapter
from nebula_model import NEBULA_DOC_MODEL, NEBULA_WORD_MODEL
from arango import ArangoClient




# Specify attributes to be imported
def nebula_get_graph_formdb(ma, _filter):
    attributes = { 'vertexCollections':
                                    {'Actors': {'labels','description'},'Actions': {'labels','description'},
                                    'Relations': {'labels','description'}, 'Properties':{'labels','description'}},
                    'edgeCollections' :
                                    {'ActorToAction': {'_from', '_to','labels'},'ActorToRelation': {'_from', '_to','labels'}, 
                                    'MovieToActors':{'_from', '_to', 'labels'}, 'RelationToProperty':{'_from', '_to', 'labels'}}}

    # Export networkX graph  
    _filter = _filter                     
    g, lables, descriptions = ma.create_nebula_graph(graph_name = 'Test',  graph_attributes = attributes, _filter = _filter)
    fitG = nx.convert_node_labels_to_integers(g, first_label=0)
    return fitG, lables, descriptions

def nebula_get_stories(all_movies,ma):
    story = 0
    documents = []
    tags = {}
    nebula_metadata = {}
    for movie in all_movies.values():
        #print(movie['movie']['_id'], movie['movie']['movie_id'], movie['movie']['file_name'])
        fitG, lables, descriptions = nebula_get_graph_formdb(ma, movie['movie']['movie_id'])
        node = 0
        _prefix = ''.join([i for i in descriptions[node][0] if not i.isdigit()])
        #if _prefix == "person":
        prefix_labels = [] 
        stories = []
        #print(fitG.nodes[node]['attr_dict']['_class'])
        if fitG.nodes[node]['attr_dict']['_class'] == "person":
            stories.append(fitG.nodes[node]['attr_dict']['_object'])
            stories = stories + lables[node] 
        else:
            stories.append(fitG.nodes[node]['attr_dict']['_class'])
        neb_feature_prev = ""
        for neb in nx.dfs_predecessors(fitG, node, 1000):
            # print(fitG.nodes[neb]['attr_dict']['_class'] )
            # print(fitG.nodes[neb]['attr_dict']['_object'])
            # input("Press Enter to continue...")
            if fitG.nodes[neb]['attr_dict']['_class'] == "person":
                neb_feature = str(fitG.nodes[neb]['attr_dict']['_object'])
                prefix_labels =lables[neb]
            else:
                neb_feature = str(fitG.nodes[neb]['attr_dict']['_class'])
                prefix_labels = []
            if neb_feature_prev != neb_feature:
                stories.append(neb_feature)
                stories = stories + prefix_labels
                neb_feature_prev = neb_feature
            else:
                neb_feature_prev = neb_feature
                continue
            #print(node, descriptions[node], prefix_labels)
            #print(movie)
            #_tag =  _prefix + "_" + str(story) + "_" + movie['movie']['file_name']
        _tag =  "story_" + str(story)
        
        #print("TAG: ", _tag)    
        #dfs_doc = TaggedDocument(words= dfs_nebs + prefix_labels, tags=[_tag])
        dfs_doc = TaggedDocument(words= stories, tags=[_tag])
        
        # print(movie['movie']['_id'])
        # print("DFS-based neibs")
        # print(_tag)
        # print(dfs_doc)
        # input("Press Enter to continue...")
        documents.append(dfs_doc)
        tags[story]= _tag
        nebula_metadata[story] = (movie['movie']['file_name'], movie['movie']['_id'], _prefix, descriptions[node][0], descriptions[node][1])
        story = story + 1
    print("Number of stories:", story)
    return(documents, tags, nebula_metadata)

def nebula_get_sentence(all_movies,ma):
    story = 0
    documents = []
    tags = {}
    nebula_metadata = {}
    for movie in all_movies.values():
        #print(movie['movie']['_id'], movie['movie']['movie_id'], movie['movie']['file_name'])
        fitG, lables, descriptions = nebula_get_graph_formdb(ma, movie['movie']['movie_id'])
        node = 0
        _prefix = ''.join([i for i in descriptions[node][0] if not i.isdigit()])
        #if _prefix == "person":
        prefix_labels = [] 
        stories = [movie['movie']['file_name'],movie['movie']['_id']]
        #stories = stories.append(movie['movie']['file_name'])
        #print(fitG.nodes[node]['attr_dict']['_class'])
        if fitG.nodes[node]['attr_dict']['_class'] == "person":
            stories.append(fitG.nodes[node]['attr_dict']['_object'])
            stories = stories + lables[node] 
        else:
            stories.append(fitG.nodes[node]['attr_dict']['_class'])
        neb_feature_prev = ""
        for neb in nx.dfs_predecessors(fitG, node, 1000):
            # print(fitG.nodes[neb]['attr_dict']['_class'] )
            # print(fitG.nodes[neb]['attr_dict']['_object'])
            # input("Press Enter to continue...")
            if fitG.nodes[neb]['attr_dict']['_class'] == "person":
                stories.append(movie['movie']['_id'])
                neb_feature = str(fitG.nodes[neb]['attr_dict']['_object'])
                prefix_labels =lables[neb]
            else:
                neb_feature = str(fitG.nodes[neb]['attr_dict']['_class'])
                prefix_labels = []
            if neb_feature_prev != neb_feature:
                stories.append(neb_feature)
                stories = stories + prefix_labels
                neb_feature_prev = neb_feature
                #print(stories)
            else:
                neb_feature_prev = neb_feature
                continue
        _tag =  "story_" + str(story)
        
        # print(movie['movie']['_id'])
        # print("DFS-based neibs")
        # print(_tag)
        # print(stories)
        # input("Press Enter to continue...")
        documents.append(stories)
        tags[story]= _tag
        nebula_metadata[story] = (movie['movie']['file_name'], movie['movie']['_id'], _prefix, descriptions[node][0], descriptions[node][1])
        story = story + 1
    print("Number of centences:", story)
    return(documents, tags, nebula_metadata)
    
def create_doc_embeddings(stories, _tags):    
    #print("DEBUG ",nebula_meta)
    #input("Press Enter to continue...")
    model = NEBULA_DOC_MODEL(dimensions=100, epochs=50)
    model.fit(stories, _tags)
    embeddings = model.get_embedding()
    return(embeddings)

def create_word_embeddings(sentences, nebula_meta):
    #print("DEBUG ",sentences)
    #input("Press Enter to continue...")
    model = NEBULA_WORD_MODEL(dimensions=100, epochs=50)
    model.fit(sentences, nebula_meta)
    embeddings = model.get_embedding()
    return(embeddings)
   
def save_embeddins(db, embeddings,nebula_meta, story, algo):
    embedding_col = db.collection('Embedding')
    for i, embedding in enumerate(embeddings):
        embedding_col.insert(
            {
                'class': nebula_meta[i][2],
                'actor_id': nebula_meta[i][4],
                'actor_name': nebula_meta[i][3],
                'movie_id':nebula_meta[i][1],
                'movie_name': nebula_meta[i][0],
                'embeddings': embedding.tolist(),
                'story': story[i],
                'algo': algo
            })

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

def main():
    # Specify the connection to the ArangoDB Database
    con = {'dbName': 'nebula_dev',
    'username': 'nebula',
    'password': 'nebula',
    'hostname': '18.159.140.240',
    'protocol': 'http', 
    'port': 8529}

    # Create Adapter instance
    ma = Nebula_Networkx_Adapter(conn = con) 
    db = connect_db('nebula_dev')
    if db.has_collection('Embedding'):
        db.delete_collection('Embedding')
    db.create_collection('Embedding')
    all_movies = ma.nebula_get_all_movies()
    stories, _tags, nebula_meta = nebula_get_stories(all_movies, ma)
    centences, _tags_cent, nebula_meta_cent = nebula_get_sentence(all_movies, ma)
    embeddings_doc = create_doc_embeddings(stories, _tags) 
    embeddings_word = create_word_embeddings(centences, nebula_meta_cent)
    save_embeddins(db, embeddings_doc,nebula_meta, stories, "NEBULA_DOC")  
    save_embeddins(db, embeddings_word,nebula_meta, centences, "NEBULA_WORD")  

if __name__ == '__main__':
    main()
