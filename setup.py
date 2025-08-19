import sys
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# Sets up the GloVe text file as Keyed Vectors by converting first to Vectors using Word2Vec.
if __name__ == "__main__":
    try:
        base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        
        api_path = os.path.join(base_path, "api")
        
        print(f"Opening {sys.argv[1]} to output to output.txt")

        glove_input_file = sys.argv[1] #"FT_bins/glove.42B.300d.txt"
        
        word2vec_output_file = "output.txt" #"FT_bins/glove.42B.300d.word2vec.txt"

        print('Step 1 of 2: Converting GloVe to word2vec format...')
        glove2word2vec(glove_input_file, word2vec_output_file)
        print('Conversion complete.')

        print('Step 2 of 2: Storing word2Vec as Keyed Vectors in \\api\\glove_vectors.kv...')
        glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        glove.save(os.path.join(api_path, "glove_vectors.kv"))
        print('Keyed Vectors stored properly.')

        print('Program finished. Closing...')

    
    except(IndexError):
        print("Invalid inputs. Terminating program.")