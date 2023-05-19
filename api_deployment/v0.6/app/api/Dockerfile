# base on NGC
FROM nvcr.io/nvidia/pytorch:22.04-py3
# install script
COPY ./app/api/ ./app/api/
RUN pip install spacy_langdetect==0.1.2 pycountry==22.3.5 stanza==1.3.0 transformers==4.17.0 transformers[sentencepiece]==4.17.0
RUN python ./app/api/install_modules.py 

