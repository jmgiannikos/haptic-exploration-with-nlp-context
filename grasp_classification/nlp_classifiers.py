import torch.nn as nn
import torch
import clip

import nlp_cls_pipeline_configs as configs

class Nlp_classifier2(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip, _ = clip.load("ViT-L/14", device=configs.get_device())

        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(192, 5),
            nn.Softmax(1)
        )

        self.name = "nlp cls v2"

    def forward(self, prompt):
        prompt_tokens = clip.tokenize(prompt).to(configs.get_device())
        prompt_embedding = self.clip.encode_text(prompt_tokens)
        embedding = self.classifier(prompt_embedding.float())
        return embedding  

class Nlp_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip, _ = clip.load("ViT-B/32", device=configs.get_device())

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Softmax(1)
        )

        self.name = "nlp cls v1"

    def forward(self, prompt):
        prompt_tokens = clip.tokenize(prompt).to(configs.get_device())
        prompt_embedding = self.clip.encode_text(prompt_tokens)
        embedding = self.classifier(prompt_embedding.float())
        return embedding    