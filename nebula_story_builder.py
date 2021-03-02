from sys import prefix
from gensim.models.doc2vec import TaggedDocument

from multiprocessing.managers import all_methods
import networkx as nx
import numpy as np
from numpy.lib.function_base import average
from scipy.ndimage.measurements import label
from scipy.stats.stats import mode
import sentence_transformers
from nebula_networkx_adapter import Nebula_Networkx_Adapter
from nebula_model import NEBULA_DOC_MODEL, NEBULA_WORD_MODEL
from arango import ArangoClient
from networkx.algorithms import centrality, community

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

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
        fitG, lables, descriptions = nebula_get_graph_formdb(ma, movie['movie']['movie_id'])
        prefix_labels = [] 
        stories = []
        story_thread = ""
        pois = []
        _count = fitG.number_of_nodes()
        for node in fitG.nodes():
            if node == 0:
                pois.append(node)
            else:
                neigh = len(list(nx.neighbors(fitG, node)))
                if neigh > 3:
                   pois.append(node)
            # else:
            #     _count = _count - 1      
       
        successors = list(nx.dfs_preorder_nodes(fitG)) 
    
        for successor in successors:
            #print(fitG.nodes[successor]['attr_dict']['_class'], successor)
            nebs = dict(nx.bfs_successors(fitG, successor))
            #print(nebs)
            stories.append(fitG.nodes[successor]['attr_dict']['_object'])
            for neb in nebs[successor]:
                #print("    ",fitG.nodes[neb]['attr_dict']['_class'], neb)
                stories.append(fitG.nodes[successor]['attr_dict']['_class'] + "_" + fitG.nodes[neb]['attr_dict']['_class'])
            
            
            
            _tag =  "story_" + str(story)
            dfs_doc = TaggedDocument(words= stories, tags=[_tag])
        
        # print(movie['movie']['_id'])
        # print("DFS-based neibs")
        # print(_tag)
        # print (len(stories))
        # print(dfs_doc)
        # input("Press Enter to continue...")
        documents.append(dfs_doc)
        tags[story]= _tag
        nebula_metadata[story] = (movie['movie']['file_name'], movie['movie']['_id'])
        print(nebula_metadata[story], story)
        story = story + 1
        
    print("Number of stories:", story)
    #print(documents)
    return(documents, tags, nebula_metadata)

def save_stories(db, nebula_meta, story):
    stories_col = db.collection('Stories')
    for i in nebula_meta:
        stories_col.insert(
            {
                'movie_id':nebula_meta[i][1],
                'movie_name': nebula_meta[i][0],
                'story': story[i]
            })


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
    if db.has_collection('Stories'):
        db.delete_collection('Stories')
    db.create_collection('Stories')
    all_movies = ma.nebula_get_all_movies()
    stories, _tags, nebula_meta = nebula_get_stories(all_movies, ma)
    save_stories(db, nebula_meta, stories)  

if __name__ == '__main__':
    main()