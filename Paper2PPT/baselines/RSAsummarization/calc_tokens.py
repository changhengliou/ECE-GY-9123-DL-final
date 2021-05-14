import os

DATAPATH = '../../../data/test'

TEST_PAPER = {
    'future': 'future/test.txt.src',
    'contribution': 'contribution/test.txt.src',
    'baseline': 'baseline/test.txt.src',
    'dataset': 'dataset/test.txt.src',
}

def paper_iterator():
    for topic in TEST_PAPER.keys():
        src_file = os.path.join(DATAPATH, TEST_PAPER[topic])

        with open(src_file, 'r') as stream:
            raw_papers = stream.readlines()
        papers_sent = [[sent.strip() for sent in raw_paper.split('##SENT##')] for raw_paper in raw_papers]
        papers_word = [[word for sent in raw_paper.split('##SENT##') for word in sent.strip().split()] for raw_paper in raw_papers]

        print(topic)
        print('Max sent length', len(max(papers_sent, key=len)))
        # print('Max sent', max(papers_sent, key=len))
        print('Max word length', len(max(papers_word, key=len)))
        # print('Max word', max(papers_word, key=len))
        import ipdb; ipdb.set_trace()


paper_iterator()

# 7499
