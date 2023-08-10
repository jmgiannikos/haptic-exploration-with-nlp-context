import clip
import numpy as np
import json
import torch

import grasp_cls_pipeline_configs as configs

LOAD_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_use_dataset.npz"
SAVE_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_use_embeddings.json"

def main():
    clip_model_l, _ = clip.load("ViT-L/14", device='cpu')
    clip_model_b, _ = clip.load("ViT-B/32", device="cpu")

    language_prompts = np.load(LOAD_PATH)

    result_dict = {}
    for key in language_prompts.keys():
        prompt_list = language_prompts[key]
        for i, prompt in enumerate(prompt_list):
            tokenized_prompt = clip.tokenize(prompt).to("cpu")
            embedding_l = list(torch.squeeze(clip_model_l.encode_text(tokenized_prompt).double()).detach().numpy())
            embedding_b = list(torch.squeeze(clip_model_b.encode_text(tokenized_prompt).double()).detach().numpy())
            prompt_and_embeddings = [prompt, embedding_l, embedding_b]
            if i == 0:
                result_dict[key] = [prompt_and_embeddings]
            else:
                result_dict[key].append(prompt_and_embeddings)

    out_file = open(SAVE_PATH, "w")
  
    json.dump(result_dict, out_file)

if __name__ == '__main__':
    main()