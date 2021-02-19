#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz


import numpy as np
import argparse
import json
import csv
import os
# import time

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCT = {
    '#', '$', '.', ',', ':', '(', ')', '"', '‘', '“', '’', '”'}
# ALT = {}
# CENTER = {}
# LEFT = {}
# RIGHT = {}

def extract1(comment):
    """ This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    """
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    cap = 0
    mod_comment = ""
    comment_split = comment.split()
    for token in comment_split:  # feature 1: upper case count
        if token[0].isupper():
            cap += 1
        text_index = token.find("/")
        mod_comment += token[:text_index].lower()
        mod_comment += token[text_index:]
        mod_comment += " "

    # TODO: Extract features that do not rely on capitalization.
    mod_comment_split = mod_comment.split()
    first, second, third = 0, 0, 0
    cc, cn, pn, adv, wh = 0, 0, 0, 0, 0
    pt, ft = verb_tense(mod_comment)  # features 6, 7: past and future tense
    comma, multichar = 0, 0
    slang = 0
    token_count, len_token = 0, 0
    aoa = []
    img = []
    fam = []
    valence = []
    arousal = []
    dominance = []
    for token in mod_comment_split:
        text, tag = separate_token(token)
        if text in FIRST_PERSON_PRONOUNS:  # feature 2: first-person pron
            first += 1
        elif text in SECOND_PERSON_PRONOUNS:  # feature 3: second-person pron
            second += 1
        elif text in THIRD_PERSON_PRONOUNS:  # feature 4: third-person pron
            third += 1
        elif text in SLANG:  # feature 14: slang
            slang += 1

        punct = punctuation(text, tag)
        comma += punct[0]  # feature 8: comma
        multichar += punct[1]  # feature 9: multichar punct

        tag_features = tag_feature(tag)
        cc += tag_features[0]  # feature 5: coordinating conjunction
        cn += tag_features[1]  # feature 10: common noun
        pn += tag_features[2]  # feature 11: proper noun
        adv += tag_features[3]  # feature 12: adverb
        wh += tag_features[4]  # feature 13: wh- words

        if text in BGL:
            aoa.append(BGL[text][0])
            img.append(BGL[text][1])
            fam.append(BGL[text][2])

        if text in WARRINGER:
            valence.append(WARRINGER[text][0])
            arousal.append(WARRINGER[text][1])
            dominance.append(WARRINGER[text][2])

        if tag not in PUNCT:
            token_count += 1
            len_token += len(text)

    sent_count, avg_len_sent = sentence_feature(comment)  # features 17, 15
    avg_len_token = len_token / token_count if token_count > 0 else 0
    # feature 16: avg length of token

    aoa_avg, aoa_std = avg_std(aoa)  # features 18, 21
    img_avg, img_std = avg_std(img)  # features 19, 22
    fam_avg, fam_std = avg_std(fam)  # features 20, 23

    valence_avg, valence_std = avg_std(valence)  # features 24, 27
    arousal_avg, arousal_std = avg_std(arousal)  # features 25, 28
    dominance_avg, dominance_std = avg_std(dominance)  # features 26, 29

    features = [cap, first, second, third, cc, pt, ft, comma, multichar, cn,
                pn, adv, wh, slang, avg_len_sent, avg_len_token, sent_count,
                aoa_avg, img_avg, fam_avg, aoa_std, img_std, fam_std,
                valence_avg, arousal_avg, dominance_avg, valence_std,
                arousal_std, dominance_std]
    feats = np.pad(features, (0, 173 - 29))
    return feats


def avg_std(data):
    """Return the average and standard deviation of the data.

    :param data: lst, a list of numbers
    :return:
        avg: the average of the data
        std: the standard deviation of the data
    """
    data_filter = [float(i) for i in data if isfloat(i)]
    avg = np.mean(data_filter) if len(data_filter) > 0 else 0
    std = np.std(data_filter) if len(data_filter) > 0 else 0
    return avg, std


def isfloat(value):
    try:
        i = float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_bgl(path):
    """Return the extracted data from the file in a dictionary.

    :param path: str, the path to the BGL file
    :return: bgl: dic, dictionary storing the extracted data
    """
    bgl = {}
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # extract field names
        fields = next(reader)
        # extract data from each row
        for row in reader:
            token = row[1]
            aoa = row[3]
            img = row[4]
            fam = row[5]
            features = [aoa, img, fam]
            bgl[token] = features
    return bgl


def load_warringer(path):
    """Return the extracted data from the file in a dictionary.

    :param path: str, the path to the Warringer file
    :return: warringer: dic, dictionary storing the extracted data
    """
    warringer = {}
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # extract field names
        fields = next(reader)
        # extract data from each row
        for row in reader:
            token = row[1]
            valence = row[2]
            arousal = row[5]
            dominance = row[8]
            features = [valence, arousal, dominance]
            warringer[token] = features
    return warringer


def sentence_feature(comment):
    """Return the sentence features of the comment.

    :param comment: string, the body of a comment in lowercase
    :return:
        sentence_count: int, the number of sentences in the comment
        avg_len_sent: int, the average length of a sentence in the comment
    """
    sentence_count = comment.count('\n')
    if len(comment) > 0 and comment[-1] != '\n':
        sentence_count += 1
    n = sentence_count
    sentence_len = 0
    while n != 0:
        s = comment.find('\n')
        if 0 < s < len(comment) - 1:
            sub = comment[:s + 1]
        else:
            sub = comment
        sub_split = sub.split()
        sentence_len += len(sub_split)
        comment = comment[s + 1:]
        n -= 1
    avg_len_sent = sentence_len / sentence_count if sentence_count > 0 else 0
    return sentence_count, avg_len_sent


def verb_tense(comment):
    """Return the number of future tense verbs in the comment.

    :param comment: string, the body of a comment in lowercase
    :return:
        past: int, the number of past tense verbs in the comment
        future: int, the number of future tense verbs in the comment
    """
    comment_split = comment.split()
    past, future = 0, 0
    for i in range(len(comment_split)):
        token = comment_split[i]
        text, tag = separate_token(token)
        if tag == "VBD":
            past += 1
        if token == "will/MD":  # count "will"
            future += 1
        if token == "go/VBG" and i + 2 < len(comment_split):
            token_n = comment_split[i + 1]
            token_nn = comment_split[i + 2]
            text_nn, tag_nn = separate_token(token_nn)
            if token_n == "to/TO" and tag_nn == "VB":
                future += 1
    return past, future


def punctuation(text, tag):
    """Return if the token is a comma or a multi-character punctuation.

    :param
        text: string, the text part of the token
        tag: string, the tag of the token
    :return: punc: list, [comma, multi-character punctuation]
    """
    if text == ',' and tag == ',':
        return [1, 0]
    elif len(text) > 1 and tag in PUNCT:  # length > 1 and token.pos_ is PUNCT
        return [0, 1]
    return [0, 0]


def tag_feature(tag):
    """Return if the tag of the token belongs to any target features.

    :param tag: string, the tag of the token
    :return:
    """
    adv = {'RB', 'RBR', 'RBS'}
    wh = {'WDT', 'WP', 'WP$', 'WRB'}
    cc_feature, cn_feature, pn_feature, adv_feature, wh_feature = 0, 0, 0, 0, 0
    if tag == 'CC':
        cc_feature = 1
    elif tag == 'NN' or tag == 'NNS':
        cn_feature = 1
    elif tag == 'NNP' or tag == 'NNPS':
        pn_feature = 1
    elif tag in adv:
        adv_feature = 1
    elif tag in wh:
        wh_feature = 1
    return [cc_feature, cn_feature, pn_feature, adv_feature, wh_feature]


def separate_token(token):
    """Return the text and the tag of the token.

    :param
        token: string, a token in the form of text/tag
    :return:
        text: string, the text of the token
        tag: string, the tag of the token
    """
    index = token.find("/") if token.count("/") == 1 else token.rfind("/")
    text = token[:index]
    tag = token[index + 1:]
    return text, tag


def extract2(feats, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    """
    if comment_class == "Alt":
        liwc_feat = ALT[comment_id]
    elif comment_class == "Center":
        liwc_feat = CENTER[comment_id]
    elif comment_class == "Left":
        liwc_feat = LEFT[comment_id]
    elif comment_class == "Right":
        liwc_feat = RIGHT[comment_id]
    else:
        raise TypeError("Invalid class type")

    feat_padded = np.pad(liwc_feat, (29, 0))
    feat = feats + feat_padded

    return feat


def id_to_feats():
    """Create global variables of mapping comment IDs to their LIWC features.

    :return: None
    """
    global ALT, CENTER, LEFT, RIGHT
    path = "/u/cs401/A1/feats/"
    classes = ["Alt", "Center", "Left", "Right"]
    for comment_class in classes:
        id_path = path + comment_class + "_IDs.txt"
        feats_path = path + comment_class + "_feats.dat.npy"
        id_file = open(id_path, 'r')
        feats_file = np.load(feats_path)
        d = {}
        for i, comment_id in enumerate(id_file):
            d[comment_id.rstrip()] = feats_file[i]

        id_file.close()

        if comment_class == "Alt":
            ALT = d
        elif comment_class == "Center":
            CENTER = d
        elif comment_class == "Left":
            LEFT = d
        elif comment_class == "Right":
            RIGHT = d


def main(args):
    # Declare necessary global variables here.
    global BGL, WARRINGER
    BGL_path = os.path.join(args.a1_dir,
                            '../Wordlists/BristolNorms+GilhoolyLogie.csv')
    WARRINGER_path = os.path.join(args.a1_dir,
                                  '../Wordlists/Ratings_Warriner_et_al.csv')
    BGL = load_bgl(BGL_path)
    WARRINGER = load_warringer(WARRINGER_path)
    id_to_feats()

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features.
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    for i in range(len(data)):
        comment_id = data[i]["id"]
        comment = data[i]["body"]
        comment_class = data[i]["cat"]
        if comment_class == "Left":
            c = 0
        elif comment_class == "Center":
            c = 1
        elif comment_class == "Right":
            c = 2
        elif comment_class == "Alt":
            c = 3
        else:
            raise TypeError("Invalid class type")

        feature = extract1(comment)
        feature = extract2(feature, comment_class, comment_id)
        feats[i] += np.append(feature, [c])

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    # start = time.time()
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output",
                        help="Directs the output to a filename of your choice",
                        required=True)
    parser.add_argument("-i", "--input",
                        help="The input JSON file, preprocessed as in Task 1",
                        required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set"
                             " to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
    # end = time.time()
    # print(end - start)
