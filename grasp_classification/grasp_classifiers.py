import torch.nn as nn
import torch

class Depth_Grasp_Classifier_v3_norm_col3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=3),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm color 3"

class Depth_Grasp_Classifier_v3_norm_col_preset_CNN(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn_feature_extract = cnn

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm color 2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_nrm_ltag_col(nn.Module):
    def __init__(self, nlp_classifier):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5189, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )
        self.nlp_model = nlp_classifier
        self.name = "v3 batch norm l-tags color"

    def forward(self, x, nlp_prompt):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        nlp_embedding = self.nlp_model(nlp_prompt)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_nrm_ltagp_col2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier1 = nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.classifier2= nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.classifier3= nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.classifier4= nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.classifier5= nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm l-tags-precalc color 2"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        x5 = self.classifier5(x)
        x = torch.cat((torch.unsqueeze(x1,2), torch.unsqueeze(x2,2), torch.unsqueeze(x3,2), torch.unsqueeze(x4,2), torch.unsqueeze(x5,2)), 2)
        x = torch.matmul(x, torch.unsqueeze(nlp_embedding,2))
        x = torch.squeeze(x)
        return x

class Depth_Grasp_Classifier_v3_norm_col2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm color 2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_norm_col(nn.Module): # BAD MODEL. ONLY REMAINS FOR REFERENCE. DO NOT USE
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm color"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_nrm_ltagp_col(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5189, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm l-tags-precalc color"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_nrm_ltagp(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5189, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm l-tags-precalc"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_norm(nn.Module): # BAD MODEL. ONLY REMAINS FOR REFERENCE. DO NOT USE
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3l(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5696, 1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 language"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5184, 1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v2_w(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1728, 864),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(864, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1728, 864),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(864, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# neural network that classifiers wether or not the grasp was successful based on the depth image
class Depth_Grasp_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 2),
            torch.nn.LogSoftmax(1)
        )
        self.name = "v1"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classifier(x)
        return x