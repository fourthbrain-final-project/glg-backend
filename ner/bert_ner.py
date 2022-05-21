#!/usr/bin/python

'''
    Created By: Brett and Jose
    Updated By: Brett and Jose
    Created On: 2022-05-19
    Updated On: 2022-05-20
    Purpose: Perform NER tasks on input documents
'''

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# tokenizers and classifiers to feed into pipeline from config files saved locally
ner_tokenizer = AutoTokenizer.from_pretrained("./models/ner/")
ner_classifier = AutoModelForTokenClassification.from_pretrained("./models/ner/")

ner_pipeline = pipeline("ner", model=ner_classifier, tokenizer=ner_tokenizer)

def get_entities(document: str) -> list:
    """
        Pass document string and return dict of entities

        Parameters
        ----------
        document: str
            document string submitted to api
        
        Returns
        -------
        : list
            list of dicts that has tagged entities
    """
    return ner_pipeline(document)


def gather_entities(entity_dict: list) -> list:
    """
        Get entities from list of type dict

        Parameters
        ----------
        entity_dict: list
            list of entities and their location in the document
        
        Returns
        --------
        : list
            list of strings that contain the entities that were found in the document                
    """
    return list(map(lambda t: t.get('word'), entity_dict))

