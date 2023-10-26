import clip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import json

import nlp_cls_pipeline_configs as configs

DATASET_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/grasp_classification/gpt_generated_object_prompt_dataset.json"

def main():
    simple_test_similarity_measure()
    #dataset_group_similarities(DATASET_PATH)

def dataset_group_similarities(path):
    model, _ = clip.load("ViT-L/14", device=configs.get_device())
    f = open(path)
    dataset = json.load(f)
    num_classes = len(list(dataset.keys()))
    similarity_matrix = np.zeros((num_classes, num_classes))
    for i, key_a in enumerate(dataset.keys()):
        for j, key_b in enumerate(dataset.keys()):
            print(f"similarity: {key_a} <-> {key_b}")
            embeddings_a = prompts_to_embeddings(dataset[key_a], model)
            embeddings_b = prompts_to_embeddings(dataset[key_b], model)
            similarity_score = generate_similarity_score(embeddings_a, embeddings_b, key_a == key_b)
            similarity_matrix[i,j] = similarity_score
    
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=dataset.keys(), yticklabels=dataset.keys())
    plt.show()

def simple_test_similarity_measure():
    eval_prompts = ["can", "cylinder", "pipe", "cube"] #["camera", "cam", "digital camera", "hair dryer", "blow drier", "microphone", " mic", "cup", "glass", "mug"]
    model, _ = clip.load("ViT-L/14", device=configs.get_device())
    for i, prompt in enumerate(eval_prompts):
        prompt_tokens = clip.tokenize(prompt).to(configs.get_device())
        prompt_embedding = model.encode_text(prompt_tokens)
        if i == 0:
            embeddings = prompt_embedding.cpu().detach().numpy()
        else:
            embeddings = np.append(embeddings, prompt_embedding.cpu().detach().numpy(), 0)
    
    cosine_similarity_matrix = cosine_similarity(embeddings, embeddings)

    sns.heatmap(cosine_similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=eval_prompts, yticklabels=eval_prompts)
    plt.show()

def prompts_to_embeddings(prompts, model):
    for i, prompt in enumerate(prompts):
        prompt_tokens = clip.tokenize(prompt).to(configs.get_device())
        prompt_embedding = model.encode_text(prompt_tokens)
        if i == 0:
            embeddings = prompt_embedding.cpu().detach().numpy()
        else:
            embeddings = np.append(embeddings, prompt_embedding.cpu().detach().numpy(), 0)
    
    return embeddings

def generate_similarity_score(embeddings_a, embeddings_b, identical=False):
    similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)
    if identical:
        similarity_score = upper_triangle_sum_no_diag(similarity_matrix)
    else:
        entry_count = np.shape(similarity_matrix)[0] * np.shape(similarity_matrix)[1]  
        similarity_score = np.sum(similarity_matrix)/entry_count

    return similarity_score

def upper_triangle_sum_no_diag(matrix):
    len = np.shape(matrix)[0]
    element_count = 0
    running_sum = 0
    idx_i = 0
    idx_j = 1

    while idx_i < len-1:
        while idx_j < len:
            element = matrix[idx_i][idx_j]
            running_sum += element
            element_count += 1
            idx_j += 1
        idx_i += 1
        idx_j = idx_i + 1

    return running_sum/element_count

def upper_triangle_sum(matrix):
    len = np.shape(matrix)[0]
    element_count = 0
    running_sum = 0
    idx_i = 0
    idx_j = 0

    while idx_i < len:
        while idx_j < len:
            element = matrix[idx_i][idx_j]
            running_sum += element
            element_count += 1
            idx_j += 1
        idx_i += 1
        idx_j = idx_i

    return running_sum/element_count

if __name__ == '__main__':
    main()
