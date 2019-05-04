import argparse
import time
import json
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output
import tensorflow as tf
from train_tf import Model

np.random.seed(1337)
random.seed(1337)


def label_rest_xml(fn, output_fn, corpus, label):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("Opinions")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("Opinion")
                    opin.attrib['target'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ':
                pass
            elif tokens[token_idx][pt:pt + 2] == '``' or tokens[token_idx][pt:pt + 2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("Opinion")
            opin.attrib['target'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn)


def label_laptop_xml(fn, output_fn, corpus, label):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("aspectTerms")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("aspectTerm")
                    opin.attrib['term'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ' or ord(c) == 160:
                pass
            elif tokens[token_idx][pt:pt + 2] == '``' or tokens[token_idx][pt:pt + 2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("aspectTerm")
            opin.attrib['term'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn)


def test(sess, model, test_X, raw_X, domain, command, template, batch_size=128, crf=False):
    pred_y = np.zeros((test_X.shape[0], 83), np.int16)

    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len = np.sum(test_X[offset:offset + batch_size] != 0, axis=1)
        batch_idx = batch_test_X_len.argsort()[::-1]
        batch_test_X_len = batch_test_X_len[batch_idx]
        batch_test_X_mask = (test_X[offset:offset + batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_test_X = test_X[offset:offset + batch_size][batch_idx]
        r_idx = batch_idx.argsort()

        fd = {model.x: batch_test_X, model.x_len: batch_test_X_len, model.dropout: 1.0}
        batch_pred_y = sess.run(model.labels_pred, feed_dict=fd)[r_idx]
        pred_y[offset:offset + batch_size, :batch_pred_y.shape[1]] = batch_pred_y

    assert len(pred_y) == len(test_X)

    command = command.split()
    if domain == 'restaurant':

        pred_y = pred_y[:, :78]
        pred_y = np.load("data/official_data/restaurants_2016_pred_labels.npy")
        print(pred_y.shape)

        label_rest_xml(template, command[6], raw_X, pred_y)
        acc = check_output(command).split()
        print(acc)
        return float(acc[9][10:])
    elif domain == 'laptop':
        label_laptop_xml(template, command[4], raw_X, pred_y)
        acc = check_output(command).split()
        print(acc)
        return float(acc[15])


def evaluate(runs, data_dir, model_dir, domain, command, template):
    gen_emb = np.load(data_dir + "gen.vec.npy")
    domain_emb = np.load(data_dir + domain + "_emb.vec.npy")
    # domain_emb = np.load(data_dir + "yelp_reviews_double.400d.txt.npy")

    ae_data = np.load(data_dir + domain + ".npz")
    with open(data_dir + domain + "_raw_test.json") as f:
        raw_X = json.load(f)
    results = []
    for r in range(runs):
        model = Model(gen_emb, domain_emb, 3, crf=False, testing=True)
        saver = tf.train.Saver()
        gpuConfig = tf.ConfigProto()
        gpuConfig.gpu_options.allow_growth = True
        with tf.Session(config=gpuConfig) as sess:
            saver.restore(sess, model_dir + domain + '_tf_' + str(r))
            result = test(sess, model, ae_data['test_X'], raw_X, domain, command, template, crf=False)
            print(result)
            results.append(result)
    print(sum(results) / len(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--model_dir', type=str, default="./model/")
    parser.add_argument('--domain', type=str, default="laptop")

    args = parser.parse_args()

    if args.domain == 'restaurant':
        command = "java -cp script/A.jar absa16.Do Eval -prd data/official_data/pred.xml -gld data/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template = "data/official_data/EN_REST_SB1_TEST.xml.A"
    elif args.domain == 'laptop':
        command = "java -cp script/eval.jar Main.Aspects data/official_data/pred.xml data/official_data/Laptops_Test_Gold.xml"
        template = "data/official_data/Laptops_Test_Data_PhaseA.xml"

    evaluate(args.runs, args.data_dir, args.model_dir, args.domain, command, template)
