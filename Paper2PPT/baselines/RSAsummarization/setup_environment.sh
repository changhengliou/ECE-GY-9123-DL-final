#!/bin/bash

# https://pip.pypa.io/en/stable/installing/
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python2 get-pip.py

sudo apt-get install -y python2.7-dev

pip install -r requirements.txt --user

python2 -c "import nltk; nltk.download('stopwords')"

# https://github.com/circulosmeos/gdown.pl
# (https://stackoverflow.com/questions/20665881/direct-download-from-google-drive-using-google-drive-api)
# (https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive)
wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
# 
perl gdown.pl https://drive.google.com/open?id=0B7pQmm-OfDv7ZUhHZm9ZWEZidDg pretrained_model_tf1.2.1.zip
unzip pretrained_model_tf1.2.1.zip

# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
perl gdown.pl https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing GoogleNews-vectors-negative300.bin.gz
# https://unix.stackexchange.com/questions/156261/unzipping-a-gz-file-without-removing-the-gzipped-file
gunzip GoogleNews-vectors-negative300.bin.gz

echo "Get vocab"
# https://github.com/abisee/cnn-dailymail
# https://stackoverflow.com/questions/1078524/how-to-specify-the-location-with-wget
mkdir finished_files
wget https://github.com/shivam13juna/Pointer_Generator/raw/master/output/vocab -O finished_files/vocab

echo "Clean up..."
rm get-pip.py
rm gdown.pl
rm pretrained_model_tf1.2.1.zip
