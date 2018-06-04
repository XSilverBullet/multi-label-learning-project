

import numpy as np
import os
base_path = os.path.join("../","../suning_data/")
word_path = os.path.join(base_path, "sgns.baidubaike.bigram-char")
sentence_path = os.path.join(base_path, "data.txt")


if __name__=="__main__":
    dictionary={}
    print("start load word2vec...")
    with open(word_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0:
                temp = str(line).rstrip('\n').split(" ")
                dictionary[temp[0]] = temp[1:-1]
    print("end...")

    print("start cal paragraph vector...")

    with open("mean_word2vec1.txt", "w", encoding="utf-8") as file:
        with open(sentence_path, encoding="utf-8") as f:


            for i, line in enumerate(f):
                sentence = str(line).rstrip("\n").split(" ")
                all_para_word2vect = []

                len = 0

                for word in sentence:
                    len += 1
                    if dictionary.__contains__(word):

                        word2vec = dictionary[word]
                        word2vec = [float(vec) for vec in word2vec]
                        all_para_word2vect.append(np.array(word2vec))
                    else:
                        word2vec = np.zeros(shape=300)
                        all_para_word2vect.append(word2vec)
                all_para_word2vect = np.asarray(all_para_word2vect, dtype=float)
                #print(all_para_word2vect.shape)
                sum_word2vec = np.sum(all_para_word2vect, axis=0)
                #print(sum_word2vec.shape)
                mean_word2vec = sum_word2vec/float(len)
                print(i)
                sentence_word2vec=""
                for vect in mean_word2vec:

                    sentence_word2vec = sentence_word2vec + str(vect)+" "
                file.write(sentence_word2vec + "\n")

    print("end...")



