import pickle
import codecs
import numpy as np
import utils


def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            # id_freqs = line.split()[1:]
            doc = {}
            count = 0
            for id_freq in id_freqs:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                # doc[int(items[0])] = int(items[1])
                count += int(items[1])
            if count > 0:
                data_list.append(doc)
                word_count.append(count)

    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.float)
    for doc_idx, doc in enumerate(data_list):
        for word_idx, count in doc.items():
            data_mat[doc_idx, word_idx] += count

    return data_mat


def compute_coherence(doc_word, topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            for l in range(n + 1, N):
                p_n = 0.0
                p_l = 0.0
                p_nl = 0.0
                for j in range(doc_size):
                    if doc_word[j, word_array[n]] != 0:
                        p_n += 1
                    # whether l^th top word in doc j^th
                    if doc_word[j, word_array[l]] != 0:
                        p_l += 1
                    # whether l^th and n^th top words both in doc j^th
                    if doc_word[j, word_array[n]] != 0 and doc_word[j, word_array[l]] != 0:
                        p_nl += 1
                if p_n > 0 and p_l > 0 and p_nl > 0:
                    p_n = p_n / doc_size
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score


def print_coherence(model='cr_nvctm', url='./data/Snippets/train.feat', vocab_size=30642):
    with codecs.open('./{}_train_beta'.format(model), 'rb') as fp:
        beta = pickle.load(fp)
    fp.close()

    test_mat = data_set(url, vocab_size)

    top_n = [5, 10, 15]
    coherence = 0.0
    for n in top_n:
        coherence += compute_coherence(test_mat, np.array(beta), n)
    coherence /= len(top_n)

    print('| NPMI score: {:f}'.format(coherence))
