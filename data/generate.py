import itertools
import csv
import os
import openai
from enum import Enum

openai.api_key = "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"

class Author(Enum):
    HUMAN = 0
    CHATGPT = 1

'''
Reads inputs one by one from read_path, rephrases each row with ChatGPT, and writes the results to write_path
If a human written or rephrased ChatGPT abstract is more then 2500 characters, then neither abstract is written
If a rephrased ChatGPT abstract is identical to the original human one, then neither abstract is written

start: index of the row to begin reading at (inclusive) in read_path csv file
stop: index of the row to stop reading at (exclusive) in read_path csv file
read_path: the path to the csv to read from, assumes each row has the columns 'terms', 'titles', 'abstracts' in that order
write_path: the path to the csv to write to, writes each abstract in the first column and whether it was human- or chatgpt-generated in the second column
mode: mode with which to write. 'w' writes from the beginning of the file, 'a' appends to the end
'''
def generate(start, stop, read_path, write_path, mode):
    with open(read_path) as read_csv, open(write_path, mode) as write_csv:
        for line in itertools.islice(csv.reader(read_csv), start, stop):
            human_abstract = line[2].replace('\n', ' ')
            if len(human_abstract) > 2500: continue
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": "Rephrase the following paragraph: "+human_abstract}
                ]
            )
            chatgpt_abstract = completion.choices[0].message.content
            if len(chatgpt_abstract) > 2500 or human_abstract == chatgpt_abstract: continue;
            csv_writer = csv.writer(write_csv)
            csv_writer.writerow([human_abstract, Author.HUMAN])
            csv_writer.writerow([chatgpt_abstract, Author.CHATGPT])


def main():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_read_path = "arxiv_abstracts.csv"
    rel_write_path = "test.csv"
    read_path = os.path.join(script_dir, rel_read_path)
    write_path = os.path.join(script_dir, rel_write_path)
    generate(601, 700, read_path, write_path, 'a')

if __name__ == "__main__":
    main()