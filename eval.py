import util
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, \
    roc_curve
from operator import itemgetter


COLORS = ['r', 'b', 'g', 'm', 'y', 'c', 'k', '#FF9900', '#006600']


def run_evaluation(examples, methods, precision_at=20):
    curve_args = []

    for i, method in enumerate(methods):
        predictions = util.load_json('./data/train/' + method + '.json')
        total_precision = 0
        all_ys, all_ps = [], []
        for u in examples:
            ys, ps = zip(*[(examples[u][b], predictions[u][b]) for b in examples[u]])
            all_ys += ys
            all_ps += ps

            n = min(precision_at, len(ys))
            top_ys = zip(*sorted(zip(ys, ps), key=itemgetter(1), reverse=True))[0][:n]
            total_precision += sum(top_ys) / float(n)

        pr_auc = average_precision_score(all_ys, all_ps)
        roc_auc = roc_auc_score(all_ys, all_ps)
        #p, r, t = precision_recall_curve(all_ys, all_ps)
        fpr, tpr, t = roc_curve(all_ys, all_ps)
        curve_args.append((fpr, tpr, method, COLORS[i]))

        print "Method:", method
        print "  Precision @{:} = {:.4f}".format(precision_at, total_precision / len(examples))
        #print "  PR Auc = {:.4f}".format(pr_auc)
        print "  ROC Auc = {:.4f}".format(roc_auc)

    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    plt.figure(figsize=(9, 9))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.title('ROC curves')
    for (fpr, tpr, label, color) in curve_args:
        plt.plot(fpr, tpr, label=label, color=color)
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    run_evaluation(util.load_json('data/train/examples.json'), ['random_baseline'])
