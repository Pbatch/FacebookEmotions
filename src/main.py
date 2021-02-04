import glob
import json
import matplotlib.pyplot as plt
from nltk import tokenize
from wordcloud import WordCloud
from emotion import Emotion
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import numpy as np
from PIL import Image
import argparse
import os


def load_sentences(path, friend, limit):
    """
    Load up to 'limit' sentences from 'path' for the given 'friend'
    """
    sentences = []
    for p in glob.glob(f'{path}/messages/inbox/{friend}_*/*.json'):
        with open(p, 'r') as f:
            data = json.load(f)
            for message in data['messages']:
                if 'content' not in message:
                    continue
                content = message['content']
                # The maximum sentence length for the transformer is 512
                tokenized_content = [s[:512] for s in tokenize.sent_tokenize(content)]
                sentences.extend(tokenized_content)
    random.shuffle(sentences)

    if len(sentences) == 0:
        raise ValueError(f'No sentences were found for friend {friend} at {path}')

    return sentences[:limit]


def group_sentences_and_labels(sentences, labels):
    """
    Keep only the sentences which have one of the following six labels:
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'
    """
    emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    emotion_dict = {emotion: [] for emotion in emotions}
    for sentence, label in zip(sentences, labels):
        if label in emotion_dict:
            emotion_dict[label].append(sentence)
    return emotion_dict


def create_wordclouds(emotion_dict):
    """
    Create six wordclouds, one for each emotion
    """
    wordclouds = []
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Load the stopwords
    stopwords_path = os.path.join(cwd, 'stopwords.txt')
    with open(stopwords_path, 'r') as f:
        stopwords = set([w.strip() for w in f])

    # Create the wordclouds
    for emotion, sentences in emotion_dict.items():
        img_path = os.path.join(cwd, f'masks/{emotion}.png')
        img = Image.open(img_path)
        mask = np.array(img.convert('RGB'))
        wc = WordCloud(background_color='whitesmoke',
                       stopwords=stopwords,
                       mask=mask,
                       contour_width=3,
                       max_words=50,
                       contour_color='black',
                       collocations=False,
                       )
        text = 'NAN' if len(sentences) == 0 else ' '.join(sentences)
        wc = wc.generate(text)
        wordclouds.append(wc)

    return wordclouds


def plot_wordclouds(wordclouds):
    """
    Plot all the wordclouds on a 2x3 grid
    """
    fig = plt.figure(figsize=(40, 60))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 3),
                     axes_pad=0.2)
    for ax, wc in zip(grid, wordclouds):
        ax.imshow(wc, interpolation='nearest', aspect='equal')
        ax.axis('off')

    plt.savefig('word_clouds.png', bbox_inches='tight')
    plt.clf()


def main(path, friend, limit):
    print('Loading sentences...')
    sentences = load_sentences(path=path,
                               friend=friend,
                               limit=limit)

    print('Inferring emotions...')
    labels = Emotion().predict(sentences)

    print('Grouping sentences and labels...')
    emotion_dict = group_sentences_and_labels(sentences, labels)

    print('Creating wordclouds...')
    wordclouds = create_wordclouds(emotion_dict)

    print('Plotting wordclouds...')
    plot_wordclouds(wordclouds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create wordclouds of your emotional interactions with a friend')
    parser.add_argument('-p', '--path',
                        help='The path to your unzipped Facebook data',
                        required=True,
                        type=str)
    parser.add_argument('-f', '--friend',
                        help='The name of a friend',
                        required=True,
                        type=str)
    parser.add_argument('-l', '--limit',
                        help='The number of messages to process',
                        default=1000,
                        type=int)
    args = parser.parse_args()
    main(args.path,
         args.friend,
         args.limit)
