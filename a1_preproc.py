import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment, steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(r"\s{2,}", " ", modComm)

    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
    if 5 in steps:
        doc = nlp(modComm)
        new_modComm = ""
        for sent in doc.sents:
            for token in sent:
                pos = token.tag_
                text = token.text
                if token.lemma_[0] == "-" and text[0] != "-":  # keep the token
                    lemma = text
                else:  # replace token with token.lemma_
                    lemma = token.lemma_
                if text.isupper():
                    lemma = lemma.upper()
                else:
                    lemma = lemma.lower()
                new_token = lemma + "/" + pos
                # add newline between each sentence
                if pos == ".":
                    new_token += "\n"
                else:
                    new_token += " "
                new_modComm += new_token

        new_modComm = new_modComm.strip("\n")
        new_modComm += "\n"
        modComm = new_modComm

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            start = args.ID[0] % len(data)
            for i in range(args.max):
                line = data[(start + i) % len(data)]
                j = json.loads(line)
                body_pp = preproc1(j["body"])
                body = body_pp
                result = {"id": j["id"], "body": body, "cat": file}
                allOutput.append(result)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output",
                        help="Directs the output to a filename of your choice",
                        required=True)
    parser.add_argument("--max", type=int,
                        help="The maximum number of comments to read "
                             "from each file", default=10000)
    parser.add_argument("--a1_dir",
                        help="The directory for A1. Should contain subdir data."
                             " Defaults to the directory for A1 on cdf.",
                        default='/u/cs401/A1')

    args = parser.parse_args()

    if args.max > 200272:
        print( "Error: If you want to read more than 200,272 comments per file,"
               " you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
