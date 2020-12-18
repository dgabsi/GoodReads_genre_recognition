# 701project
Task :Recognising book genre- based on goodreads dataset.
We will use different features-including text features(description and title), image features(Book covers)
and other numerical features(such as number of pages, ratings and more) in order to recognise a book genre.

# Important: Data files are in github (under data directory)


### github- dgabsi/701project
(updates were made from danielaneuralx which is my working github but its all mine.)

Main notebooks are:
Please run them in that order(because the first preporcess the data)
- eda_goodreads.ipynb (include exploration of dataset)
- inference_goodreads.ipynb (learning models)

Report:
701projectreport_bookgenre.pdf

Project structure:
- root
  - data
    - images-source(Directory-don't delete -this holds dataset images)
    - images-train (Directory-This directory and content will be created)
    - images-val(Directory-This directory and content will be created) 
    - images-test(Directory-This directory and content will be created)
    - goodreads_books_eng_f1.csv (Dont delete-This is the first dataset csv)
    - goodreads_books_eng_f2.csv (Dont delete-This is the second dataset csv)
  - goodreads (package)
    - _init_.py
    - baseline.py
    - conv_goodreads.py
    - custom_nn_with_ebeddings
    - results_utils.py
    - utils.py
  - configuration.yml  (Very important -dont delete)= this holds hyperparameters configuration and general parameters
  - eda_goodreads.ipynb
  - inference_goodreads.ipynb
  
Needs packages:
- tensorflow
- matplotlib
- sckit-learn
- os
- pyyaml
- numpy
- pandas
- tesnorflow-addons (for F1 metric)
- nltk
- gensim
- spacy
- nltk
    
    
should run :
python -m spacy download en_core_web_md

Please for any problem or question-find me at Daniela.Stern-Gabsi@city.ac.uk
