import util
import random


def main():
<<<<<<< HEAD
    examples = util.load_json('./data/train/examples.json')
    for u in examples:
        for b in examples[u]:
            examples[u][b] = random.random()
    util.write_json(examples, './data/train/random_baseline.json')
=======
    examples = util.load_json('./data/test/examples.json')
    for u in examples:
        for b in examples[u]:
            examples[u][b] = random.random()
    util.write_json(examples, './data/test/random_baseline.json')
>>>>>>> FETCH_HEAD

if __name__ == '__main__':
    main()
