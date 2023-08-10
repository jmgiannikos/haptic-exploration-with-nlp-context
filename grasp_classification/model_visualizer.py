from grasp_classifiers import *
import torchviz as tv


if __name__ == '__main__':
    model = Depth_Grasp_Classifier_v3_norm_col()
    image = torch.randn([2,6,80,60])
    output = model(image)
    tv.make_dot(output, params=dict(list(model.named_parameters()))).render("baseline classifier structure", format="png")