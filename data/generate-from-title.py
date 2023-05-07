import itertools
import csv
import os
from hugchat import hugchat
from enum import Enum
import threading
import math

class Author(Enum):
    HUMAN = 0
    CHATGPT = 1


# @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(10), reraise=True)
def completion_with_backoff(abstract_title):
    chatbot = hugchat.ChatBot()
    response = chatbot.chat("Write an abstract around 150 to 300 words long for a paper titled "+abstract_title+". Do not include the paper title in the abstract. Do not start with the word this.")
    return response


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
    total_skipped = 0
    total_written = 0
    try:
        with open(read_path) as read_csv, open(write_path, mode) as write_csv:
            for line in itertools.islice(csv.reader(read_csv), start, stop):
                abstract_title = line[1].replace('\n', ' ').replace('<', '').replace('>', '')
                human_abstract = line[2].replace('\n', ' ').replace('<', '').replace('>', '')
                if len(human_abstract) > 2500: 
                    print("human abstract too long (", len(human_abstract), " characters), skipping")
                    total_skipped += 1
                    continue
                success = False
                try:
                    chatgpt_abstract = completion_with_backoff(abstract_title)
                    success = True
                except Exception as e:
                    print("Request errored:", e)
                    total_skipped += 1
                if not success: continue
                chatgpt_abstract = chatgpt_abstract.replace("\n", " ")
                if len(chatgpt_abstract) > 2500:
                    print("chatgpt abstract too long (", len(chatgpt_abstract), " characters), skipping")
                    total_skipped += 1
                    continue
                if human_abstract == chatgpt_abstract: 
                    print("rephrased abstract identical to original, skipping")
                    total_skipped += 1
                    continue
                csv_writer = csv.writer(write_csv)
                csv_writer.writerow([human_abstract, Author.HUMAN.value])
                csv_writer.writerow([chatgpt_abstract, Author.CHATGPT.value])
                total_written += 1
        print("total skipped:", total_skipped)
        print("total written:", total_written)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stats so far:")
        print("total skipped:", total_skipped)
        print("total written:", total_written)
    except Exception as e:
        print ("Exception: ", e)
        print("total skipped:", total_skipped)
        print("total written:", total_written)


def main():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_read_path = "arxiv_abstracts.csv"
    # rel_write_path = "threads/thread0.csv"
    read_path = os.path.join(script_dir, rel_read_path)
    # write_path = os.path.join(script_dir, rel_write_path)

    start = 1
    num_requests = 10
    num_threads = 8
    num_requests_per_thread = math.floor(num_requests / num_threads)
    threads = []
    for i in range(num_threads):
        s = start+num_requests_per_thread*i
        e = start+num_requests if i == (num_threads-1) else s+num_requests_per_thread
        w = os.path.join(script_dir, "threads/thread"+str(i)+".csv")
        t = threading.Thread(target=generate, args=(s, e, read_path, w, 'w'))
        print("thread ",i," starting at s=",s," (incl) and going to e=",e," (excl)")
        threads.append(t)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # generate(start, start+num_requests, read_path, write_path, 'w')

if __name__ == "__main__":
    main()