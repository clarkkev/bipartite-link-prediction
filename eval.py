import util
from sklearn.metrics import precision_recall_curve, average_precision_score
from operator import itemgetter
import matplotlib.pyplot as plt


def run_evaluation(examples, predictions, precision_at=20):
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
    print "Precision @{:} = {:.4f}".format(precision_at, total_precision / len(examples))
    print "Auc = {:.4f}".format(auc)

    plt.clf()
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.title('precision-recall curve: AUC={0:0.4f}'.format(auc))
    plt.show()


if __name__ == '__main__':
    run_evaluation(util.load_json('data/train/examples.json'),
         util.load_json('./data/train/random_baseline.json'))
