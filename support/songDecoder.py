# -*- coding: utf-8 -*-
# Thin shim — all logic lives in vocalVocab.py and model.py
from support.vocalVocab import VocaloidVocab, initialize_model, tokens_from_notes
from support.model import count_parameters

initalizeModel = initialize_model  # preserve legacy spelling used in main.py
