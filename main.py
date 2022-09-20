from stylometry.extract import *
from stylometry.use_main import *


 
#cantidad de palabras que ingrean a all_authors

#esta es la linea final que se debe usar para las pruebas 
n_Crea=np.arange(50, 1001, 50) 




finalFile = 'merged.txt'

path='C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/pan15-verification-training-sp/'


pairwise_distances_methods = ['cosine', 'euclidean', 'manhattan', 'chi2', 'cityblock', 'l1', 'l2']
cdist_methods = ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'minkowski', 'sqeuclidean', 'final_array']
truth_path = 'C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/pan15-verification-training-sp/truth.txt'


for x in n_Crea:
    output_path = 'C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/salidafinal/output_' + str(x) + '.csv'

    # delete_file(path)

    scan_all_dir(path, finalFile)

    # All authors
    all_authors(path, x)

    # merging csvs
    merge_csv(output_path)

    
    add_methods(pairwise_distances_methods, cdist_methods, output_path)

    # adding truth.txt values
    truth_file(output_path, truth_path)
