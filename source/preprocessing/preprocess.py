from datasets import load_dataset
import os
import subprocess
import argparse

def zip_to_stories(zip_dir, stories_dir, min_sentence_length=10):
    '''
    Load the raw data from the zip files and save it to the stories directory

    :param zip_dir: directory containing the zip files
    :param stories_dir: directory to save the raw data to

    :return: None
    '''
    dataset = load_dataset(zip_dir+"arxiv_summarization.py", 'section')

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    for i, data in enumerate(train_dataset):
        with open(os.path.join(stories_dir+"train", "train_"+str(i)+".story"), "w") as f:
            # Write article and abstract to files, removing extra whitespaces
            f.write(data['article'].strip() + "\n\n")
            
            # We need to separate the abstract into sentences
            abstract = data['abstract'].strip().split(". ")
            abstract = [sentence for sentence in abstract if len(sentence) >= min_sentence_length]
            for sentence in abstract:
                while sentence[0] == " " or sentence[0] == "\n":
                    sentence = sentence[1:]
                    if len(sentence) < min_sentence_length:
                        break
                if len(sentence) < min_sentence_length:
                    continue
                f.write("@highlight\n\n")
                f.write(sentence + "\n\n")

    for i, data in enumerate(val_dataset):
        with open(os.path.join(stories_dir+"val", "val_"+str(i)+".story"), "w") as f:
            # Write article and abstract to files, removing extra whitespaces
            f.write(data['article'].strip() + "\n\n")
            
            # We need to separate the abstract into sentences
            abstract = data['abstract'].strip().split(". ")
            abstract = [sentence for sentence in abstract if len(sentence) >= min_sentence_length]
            for sentence in abstract:
                while sentence[0] == " " or sentence[0] == "\n":
                    sentence = sentence[1:]
                    if len(sentence) < min_sentence_length:
                        break
                if len(sentence) < min_sentence_length:
                    continue
                f.write("@highlight\n\n")
                f.write(sentence + "\n\n")

    for i, data in enumerate(test_dataset):
        with open(os.path.join(stories_dir+"test", "test_"+str(i)+".story"), "w") as f:
            # Write article and abstract to files, removing extra whitespaces
            f.write(data['article'].strip() + "\n\n")
            
            # We need to separate the abstract into sentences
            abstract = data['abstract'].strip().split(". ")
            abstract = [sentence for sentence in abstract if len(sentence) >= min_sentence_length]
            for sentence in abstract:
                while sentence[0] == " " or sentence[0] == "\n":
                    sentence = sentence[1:]
                    if len(sentence) < min_sentence_length:
                        break
                if len(sentence) < min_sentence_length:
                    continue
                f.write("@highlight\n\n")
                f.write(sentence + "\n\n")

def tokenize(args):
    '''
    Tokenize the raw data using Stanford CoreNLP

    :param args: command line arguments

    :return: None
    '''
    stories_dir = os.path.abspath(args.save_path) # Path to the folder containing the stories
    tokenized_stories_dir = os.path.abspath(args.tokenized_path) # Path to the tokenized stories

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    subdirs = os.listdir(stories_dir) # stories_dir contains subdirectories for train, val and test

    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for subdir in subdirs:
            stories = os.listdir(os.path.join(stories_dir, subdir))
            for s in stories:
                if (not s.endswith('story')):
                    continue
                f.write("%s\n" % (os.path.join(stories_dir, subdir, s))) # the input file path
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = 0

    for subdir in subdirs:
        stories = os.listdir(os.path.join(stories_dir, subdir))
        num_orig += len(stories)

    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

    # At this point, the tokenized stories directory contains story files for train, val and test.
    # Now we move them into separate directories for each dataset.
    print("Splitting the tokenized dataset into train, val and test...")
    stories = os.listdir(tokenized_stories_dir)
    for subdir in subdirs:
        if not os.path.exists(os.path.join(tokenized_stories_dir, subdir)):
            os.makedirs(os.path.join(tokenized_stories_dir, subdir))
    for s in stories:
        if (not s.endswith('story')):
            continue
        if s.startswith("train"):
            os.rename(os.path.join(tokenized_stories_dir, s), os.path.join(tokenized_stories_dir, "train", s)) # Move the file to the train folder
        elif s.startswith("val"):
            os.rename(os.path.join(tokenized_stories_dir, s), os.path.join(tokenized_stories_dir, "val", s)) # Move the file to the val folder
        elif s.startswith("test"):
            os.rename(os.path.join(tokenized_stories_dir, s), os.path.join(tokenized_stories_dir, "test", s)) # Move the file to the test folder
        else:
            raise Exception("Failed to split %s into train, val and test." % (tokenized_stories_dir))



zip_dir = "source/preprocessing/data/"
stories_dir = "source/preprocessing/data/processed/"
token_dir = "source/preprocessing/data/tokenized/"

help_message = '''
    Preprocess the raw data and tokenize it using Stanford CoreNLP

    Usage:
        python preprocess.py --raw_path <path_to_raw_data> --save_path <path_to_save_data> --tokenized_path <path_to_save_tokenized_data> --preprocess <True/False> --tokenize <True/False>
    '''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=help_message)

    argparser.add_argument('--raw_path', type=str, default=zip_dir)
    argparser.add_argument('--save_path', type=str, default=stories_dir)
    argparser.add_argument('--tokenized_path', type=str, default=token_dir)
    argparser.add_argument('--preprocess', type=bool, default=False)
    argparser.add_argument('--tokenize', type=bool, default=False)

    args = argparser.parse_args()

    if args.preprocess:
        zip_to_stories(args.raw_path, args.save_path)
    if args.tokenize:
        tokenize(args)
