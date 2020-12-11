# 701project
Task :Recognising book genre- based on goodreads dataset

Main notebooks are:
eda_goodreads.ipynb (include exploration of dataset)
inference_goodreads.ipynb (learning models)

Report:


Project structure:
-root
  -data
    -images-source(Directory-don't delete -this holds dataset images)
    -images-train (Directory-This directory and content will be created)
    -images-val(Directory-This directory and content will be created) 
    -images-test(Directory-This directory and content will be created)
    -goodreads_books_eng.csv (Dont delete-This is the dataset csv)
  -goodreads (package)
    -_init_.py
    - baseline.py
    -conv_goodreads.py
    -custom_nn_with_ebeddings
    -results_utils.py
    -utils.py
  -configuration.yml  (Very important -dont delete)= this holds hyperparameters configuration and general parameters
  -eda_goodreads.ipynb
  -inference_goodreads.ipynb
  
Needs packages:
tensorflow
matplotlib
sckit-learn
os
pyyaml
numpy
pandas
tesnorflow-addons (for F1 metric)
nltk
gensim
spacy
nltk
    
    
should run :
python -m spacy download en_core_web_md
