from registry import list_spaces
import voyageai
import numpy as np
import os

#VOYAGE_API_KEY = dbutils.secrets.get(scope="voyageai_api_key", key="api_key")
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
_space_embeddings = {}
client = voyageai.Client(api_key=VOYAGE_API_KEY)


def get_embedding(text):

    result = client.embed([text],model = 'voyage-3')
    return result.embeddings[0]


def _convert_into_embedding():

    spaces_config = list_spaces()

    for space in spaces_config:
        keywords = " ".join(space.keywords)
        space_descriptions = space.description + " " + keywords
        space_embedding = get_embedding(space_descriptions)
        _space_embeddings[space.name] = np.array(space_embedding)


_convert_into_embedding()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def detect_space(question: str):
#     question_embedding = get_embedding(question)
#     cosine_similarities = []
#     for space_name, space_embedding in _space_embeddings.items():
#         score = cosine_similarity(question_embedding, space_embedding)
#         cosine_similarities.append((space_name, score))
#     cosine_similarities.sort(key=lambda x: x[1], reverse=True)
#     print(cosine_similarities)

#     top_space, top_score = cosine_similarities[0]
#     if top_score < 0.15:
#         return None, top_score
#     return top_space, top_score


def detect_space(question: str):
    question_embedding = get_embedding(question)
    cosine_similarities = []
    for space_name, space_embedding in _space_embeddings.items():
        score = cosine_similarity(question_embedding, space_embedding)
        cosine_similarities.append((space_name, score))
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)

    top_space, top_score = cosine_similarities[0]

    ABSOLUTE_FLOOR = 0.15
    MARGIN = 0.10

    if top_score < ABSOLUTE_FLOOR:
        return None, float(top_score)

    if len(cosine_similarities) > 1:
        _, second_score = cosine_similarities[1]
        if (top_score - second_score) < MARGIN:
            return None, float(top_score)

    return top_space, float(top_score)

#print(detect_space("recommend a good movie to watch tonight"))