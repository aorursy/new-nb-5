"""

LOAD DATA

"""



import numpy as np 

import pandas as pd 

import json





train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')



train = np.array(train_df)

test = np.array(test_df)






"""

SETTINGS

"""



use_cuda = True # whether to use GPU or not
train_df.head()



"""

Prepare training data in QA-compatible format

"""



# Adpated from https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing

def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



def do_qa_train(train):



    output = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        output.append({'context': context.lower(), 'qas': qas})

        

    return output



qa_train = do_qa_train(train)



with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)



"""

Prepare testing data in QA-compatible format

"""



def do_qa_test(test):

    output = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        output.append({'context': context.lower(), 'qas': qas})

    return output



qa_test = do_qa_test(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)






from simpletransformers.question_answering import QuestionAnsweringModel



MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'



# Create the QuestionAnsweringModel

model = QuestionAnsweringModel('distilbert', 

                               MODEL_PATH, 

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 3,

                                     'max_seq_length': 192,

                                     'doc_stride': 64,

                                     'fp16': False,

                                    },

                              use_cuda=use_cuda)



model.train_model('data/train.json')



predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)



sub_df['selected_text'] = predictions_df['answer']



sub_df.to_csv('submission.csv', index=False)



print("File submitted successfully.")