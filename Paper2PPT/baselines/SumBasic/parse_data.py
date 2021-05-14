import os
import shutil
from tqdm import tqdm
import shutil

DATAPATH = '../../../data/test'
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
TARGET = 'SumBasic'

DATA_DIR = os.path.join(DATA_PATH, TARGET)
OUTPUT_DIR = os.path.join(OUTPUT_PATH, TARGET)


def parse_data():
    """ parse our annotation and write text files into DATA_DIR """

    parse = True
    if os.path.exists(DATA_DIR):
        override = input('Data exist, override (delete and re-parse)? (Y/n): ')
        if override.lower() == 'y':
            shutil.rmtree(DATA_DIR)
        else:
            parse = False
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATAPATH, 'test.txt.src'), 'r') as stream:
        raw_papers = stream.readlines()
    papers = [paper.strip().split('##SENT##') for paper in raw_papers]

    if parse:
        print('Converting src to raw text...')
        for i, paper in tqdm(enumerate(papers), total=len(papers)):

            did = f'{i+1}.txt'

            text_file = os.path.join(DATA_DIR, did)
            with open(text_file, 'w') as stream:
                # make sure the sent split are the same as our annotation
                stream.write('\n'.join(paper))


if __name__ == "__main__":
    parse_data()
