import seaborn as sns
import matplotlib.pyplot as plt

TP = 0.03725
TN = 0.93758
FP = 0.01271
FN = 0.01246

def main():
    conf_mat = [[TP,FP],[FN,TN]]
    sns.heatmap(conf_mat, annot=True, cmap="Blues", xticklabels=[1,0], yticklabels=[1,0], fmt='.4f')
    plt.xlabel('Actual Grasp Success')
    plt.ylabel('Predicted Grasp Success')
    plt.show()


if __name__ == '__main__':
    main()