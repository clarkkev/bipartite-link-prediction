import util
import random

def main():
    examples = util.load_json('./data/train/examples.json')
    for u in examples:
        for b in examples[u]:
            examples[u][b] = random.random()
    util.write_json(examples, './data/train/random_baseline.json')

if __name__ == '__main__':
    main()
