#!/usr/bin/python3

"""
    Created By: Brett and Jose
    Updated By: Brett and Jose
    Created On: 5/19/2022
    Updated On: 5/19/2022
    Purpose: Classify incoming requests as either heath/tech/other
    
"""

from transformers import pipeline

# possible classes are a constant
POSSIBLE_CLASSES = ['health', 'technology', 'other']

# loading the classifier into memory to cache it
classifier = pipeline('zero-shot-classification', model='./models/classify')


def classify_document(submitted_doc: str, classes: list = POSSIBLE_CLASSES) -> list:
    """
        Takes a text string and returns the sequence, labels, and class probabilities

        Parameters
        ----------
        submitted_doc: str
            document string to generate labels for
        
        Returns
        -------
        : list
            output from pretrained bart model
    """
    return classifier(submitted_doc, classes)

def get_class(bart_otuput: dict) -> str:
    """
        Get the class label from the return of classify document 
    """
    if 'labels' in bart_otuput:
        if isinstance(bart_otuput.get('labels'), list):
            return bart_otuput.get('labels')[0]
        else:
            return bart_otuput.get('labels')
    else:
        return ''
