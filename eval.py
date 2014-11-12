import util
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from operator import itemgetter


COLORS = ['r', 'b', 'g', 'm', 'y', 'c']


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

        auc = average_precision_score(all_ys, all_ps)
        p, r, t = precision_recall_curve(all_ys, all_ps)
        curve_args.append((p, r, method, COLORS[i]))

        print "Method:", method
        print "  Precision @{:} = {:.4f}".format(precision_at, total_precision / len(examples))
        print "  Auc = {:.4f}".format(auc)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.title('precision-recall curve: AUC={0:0.4f}'.format(auc))
    for (p, r, label, color) in curve_args:
            plt.plot(r, p, label=label, color=color)
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    run_evaluation(util.load_json('data/train/examples.json'),
         ['random_baseline', 'random_walks', 'weighted_random_walks'])
