import numpy as np
import datetime
import random
from collections import defaultdict
from scipy.misc import logsumexp


class DataSet():
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        file = open(filename, encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)  # [[word1,word2,...],[word1...],[...]]
                self.tags.append(tag)  # [[tag1,tag2,...],[tag1...],[...]]
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])  # [word1,word2,...]
                tag.append(line.split()[3])  # [tag1,tag2,...]
                word_num += 1
        self.sentences_num = len(self.sentences)  # 统计句子个数
        self.word_num = word_num  # 统计词语个数

        print('{}:共{}个句子,共{}个词。'.format(filename, self.sentences_num, self.word_num))
        file.close()

    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):  # j为词在句子中的序号
                data.append((self.sentences[i], j, self.tags[i][j]))  # [(句子1,1,词性),(句子1,2,词性)...]
        return data


class LogLinearModel(object):
    def __init__(self, train_file, dev_file, test_file):
        self.train_data = DataSet(train_file)  # 处理训练集文件
        self.dev_data = DataSet(dev_file)  # 处理开发集文件
        self.test_data = DataSet(test_file) # 处理测试集文件
        self.features = {}  # 存放所有特征及其对应编号的字典
        self.tag_dict = {}  # 存放所有词性及其对应编号的字典
        self.tag_list = []  # 存放所有词性的列表
        self.weights = []  # 特征权重矩阵
        self.g = []

    def create_feature_template(self, sentence, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])  # 第一个字符和当前字符添加到模板中
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])  # 最后一个字符和当前字符添加到模板中

        if len(sentence[position]) == 1:  # 如果当前分词只有一个字符
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, len(sentence[position]) - 1):
            if sentence[position][i] == sentence[position][i + 1]:  # 如果当前字符和下一个字符相等
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])

        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_feature_template(sentence, j)
                for f in template:  # 对特征进行遍历
                    if f not in self.features.keys():  # 如果特征不在特征字典中，则添加进去
                        self.features[f] = len(self.features)  # 给该特征一个独立的序号标记
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)
        self.tag_list = sorted(self.tag_list)
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}  # 分别列出词性下标和词性
        self.weights = np.zeros((len(self.tag_dict), len(self.features)))
        self.g = defaultdict(float)
        print("特征的总数是：{}".format(len(self.features)))

    def get_max_tag(self, sentence, position):
        f = self.create_feature_template(sentence, position)
        f_index_list = [self.features[i] for i in f if i in self.features]
        scores = np.sum(self.weights[:, f_index_list], axis=1)  # 将每个tag对应的得分加起来，得到一个得分列表
        max_tag_index = np.argmax(scores)  # 获得最大的分对应的下标
        return self.tag_list[int(max_tag_index)]

    def get_prob(self, f_index_list):
        scores = np.sum(self.weights[:, f_index_list], axis=1)  # 将每个tag对应的得分加起来，得到一个得分列表
        s = logsumexp(scores)
        prob_list = np.exp(scores-s)
        return prob_list  # 当前词标注为各词性的概率列表

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]  # 当前句子
            tags = data.tags[i]  # 当前句子的分词
            total_num += len(tags)
            for j in range(len(sentence)):
                predict_tag = self.get_max_tag(sentence, j)
                if predict_tag == tags[j]:
                    correct_num += 1
        return correct_num, total_num, correct_num / total_num

    def SGD_train(self, iteration, batch_size, shuffle, regularization, step_opt, eta, C, stop_iteration):
        b = 0
        counter = 0
        max_dev_precision = 0
        global_step = 1
        decay_steps = 100000
        decay_rate = 0.96
        learn_rate = eta
        max_iterator = 0

        data = self.train_data.split()
        print('eta={}'.format(eta))
        if regularization:
            print('使用正则化   C={}'.format(C))
        if step_opt:
            print('使用步长优化')
        for iter in range(iteration):
            print('当前迭代次数：{}'.format(iter))
            start_time = datetime.datetime.now()
            if shuffle:
                print('正在打乱训练数据...', end="")
                random.shuffle(data)
                print('数据打乱完成')
            for i in range(len(data)):
                sentence = data[i][0]
                j = data[i][1]
                gold_tag = data[i][2]
                gold_tag_index = self.tag_dict[gold_tag]
                templates = self.create_feature_template(sentence, j)
                f_index_list = [self.features[i] for i in templates if i in self.features]
                prob_list = self.get_prob(f_index_list)
                for f in f_index_list:
                    self.g[(gold_tag_index, f)] += 1
                    self.g[f, ] -= prob_list

                b += 1
                if b == batch_size:
                    if regularization:
                        self.weights *= (1 - C * learn_rate)
                    for key, value in self.g.items():
                        if len(key) == 1:  # 若关键字长度为1，即关键字不包含tag
                            self.weights[:, key] += eta * np.reshape(value, (len(self.tag_list), 1))
                        else:
                            self.weights[key] += eta * value
                    if step_opt:
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                    b = 0
                    global_step += 1
                    self.g = defaultdict(float)  # 重新初始化g

            if b > 0:
                if regularization:
                    self.weights *= (1 - C * learn_rate)
                for key, value in self.g.items():
                    if len(key) == 1:  # 若关键字长度为1，即关键字不包含tag
                        self.weights[:, key] += eta * np.reshape(value, (len(self.tag_list), 1))
                    else:
                        self.weights[key] += eta * value
                if step_opt:
                    learn_rate = eta * decay_rate ** (global_step / decay_steps)
                b = 0
                global_step += 1
                self.g = defaultdict(float)  # 重新初始化g

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：{} / {} = {}'.format(train_correct_num, total_num, train_precision))
            test_correct_num, test_num, test_precision = self.evaluate(self.test_data)
            print('\t' + 'test准确率：{} / {} = {}'.format(test_correct_num, test_num, test_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：{} / {} = {}'.format(dev_correct_num, dev_num, dev_precision))

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1
            end_time = datetime.datetime.now()
            print("\t迭代执行时间为：" + str((end_time - start_time).seconds) + " s")
            if counter >= stop_iteration:
                break
        print('最优迭代轮次 = {} , 开发集准确率 = {}'.format(max_iterator, max_dev_precision))


if __name__ == '__main__':
    train_data_file = 'data/train.conll'  # 训练集文件
    dev_data_file = 'data/dev.conll'  # 开发集文件
    test_data_file = 'data/test.conll'  # 测试集文件
    iteration = 100  # 最大迭代次数
    batch_size = 50  # 批次大小
    shuffle = True  # 每次迭代是否打乱数据
    regularization = True  # 是否正则化
    step_opt = True  # 是否步长优化,设为true步长会逐渐衰减，否则为初始步长不变
    eta = 0.5  # 初始步长
    C = 0.0001  # 正则化系数,regularization为False时无效
    stop_iteration = 10  # 连续多少次迭代没有提升效果就退出

    total_start_time = datetime.datetime.now()
    lm = LogLinearModel(train_data_file, dev_data_file, test_data_file)
    lm.create_feature_space()
    lm.SGD_train(iteration, batch_size, shuffle, regularization, step_opt, eta, C, stop_iteration)
    total_end_time = datetime.datetime.now()
    print("总执行时间为：" + str((total_end_time - total_start_time).seconds) + " s")
