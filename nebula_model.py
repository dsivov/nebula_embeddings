import random
import numpy as np
from datetime import datetime
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NEBULA_DOC_MODEL(Estimator):
    def __init__(self, dimensions: int=128, workers: int=8,
                 epochs: int=500, down_sampling: float=0,
                 learning_rate: float=0.001, min_count: int=1, seed: int=42):

        self.dimensions = dimensions
        self.workers = workers
        self.epochs = epochs
        self.down_sampling = down_sampling
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.model = None
       
    def fit(self, documents, tags):
        self._set_seed()
        print ("DOC2VEC Training")
        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=5,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        iter=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        model.save(str(timestamp) + "model_doc.dat")
        self._embedding = np.array([model.docvecs[tags[i]] for i, _ in enumerate(documents)])
        self.model = model

    def _get_embeddings(self, tags):
        embeddings = []
        for tag in tags:
            print(tag)
            embeddings.append(self.model.docvecs[tag])      
        return(np.array(embeddings))

        
    def get_embedding(self) -> np.array:
        embedding = self._embedding
        return embedding

class NEBULA_WORD_MODEL(Estimator):
    def __init__(self, dimensions: int=128, workers: int=1,
                 epochs: int=500, down_sampling: float=0.001,
                 learning_rate: float=0.01, min_count: int=1, seed: int=42):

        self.dimensions = dimensions
        self.workers = workers
        self.epochs = epochs
        self.down_sampling = down_sampling
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
       
    def fit(self, documents, meta):
        self._set_seed()
        model = Word2Vec(sentences=documents, hs=1, min_count=1,
                        size=self.dimensions, 
                        iter=self.epochs, 
                        alpha=self.learning_rate, 
                        seed=self.seed,
                        sg=1,
                        window=15)
                        #vector_size=self.dimensions,
                        #window=10,
                        #min_count=self.min_count,
                        #dm=0,
                        #sample=self.down_sampling,
                        #workers=self.workers,
                        #epoch=self.epochs,
                        #alpha=self.learning_rate,
                        #seed=self.seed)

        model.save("model_word.dat")
        vectors = []    
        for movie in meta.values():
            #print(movie[1])
            vector = model.wv[movie[1]]
            vectors.append(vector)
        self._embedding = np.array(vectors)
        #print(meta)
        #print(documents)
        #input("Press Enter to continue...")

    def get_embedding(self) -> np.array:
        embedding = self._embedding
        return embedding