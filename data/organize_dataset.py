"""
Organiza o dataset BRACOL baixado manualmente em data/raw/ nas pastas
data/processed/{train,val,test}/{classe}/, seguindo a divisão documentada
no requisito RD01 (Frente 3).

TODO (Frente 8): implementar a leitura das anotações originais do BRACOL
e a divisão estratificada por classe.
"""

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

CLASSES = ["saudavel", "ferrugem", "bicho_mineiro", "phoma", "cercosporiose"]


def organizar():
    raise NotImplementedError("Implementar na Frente 8, após o download manual do dataset.")


if __name__ == "__main__":
    organizar()