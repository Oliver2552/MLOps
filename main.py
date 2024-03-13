import argparse
from model.summarizer import Summarizer

def main(text):
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print("Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize the provided text using PEGASUS model.')
    parser.add_argument('text', type=str, help='Text to summarize')
    args = parser.parse_args()
    main(args.text)