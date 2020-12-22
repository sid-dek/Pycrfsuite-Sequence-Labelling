import hw2_corpus_tool as read_tool
import pycrfsuite as pycrf
import sys
from collections import deque
from collections import Counter

## Keeps track of number of utterances a speaker is speakingcontinuously
speaker_sentence_counter = 0

## Keeps a track of number of actual dialogs for each speaker
## (Multiple utterances by same speaker are couted as 1 utterance)
dialog_counter = 1


def advanced_features(dialog_text):
    feature = []

    ## Find the last 3 words of the dialog text
    ## According to the annotation mannual, last 3 words have an important impact on the utterance
    last1Word = "" if len(dialog_text)<1 else dialog_text[-1]
    last2Word = "" if len(dialog_text)<2 else dialog_text[-2:]
    last3Word = "" if len(dialog_text)<3 else dialog_text[-3:]

    ## Add advanced features
    feature += [
        "isEndWithHyphen={}".format(dialog_text.endswith("-/") or dialog_text.endswith("- /")),
        "isEndWithSlash={}".format(dialog_text.endswith(" /")),
        "isQuestion={}".format(dialog_text.find("?")),
        last1Word,
        last2Word,
        last3Word
    ]

    return feature

def generate_feature(dialog, isChangeSpeaker, isFirstUtterance, ngram_queue):
    global speaker_sentence_counter
    global dialog_counter

    ## Keeps a count of all the POS tags in a dialog
    pos_counter = Counter()

    ## Resets / Increments the counters
    if isChangeSpeaker:
        speaker_sentence_counter = 1
        dialog_counter += 1
    else:
        speaker_sentence_counter += 1

    ## Sets SpeakerChange to False for 1st utterance (Special Case)
    if isFirstUtterance:
        isChangeSpeaker = False
        speaker_sentence_counter = 1

    feature = []

    ## Appends all features for (n-1) previous features
    for index, ngram_feature in enumerate(ngram_queue):
        for each_feature in ngram_feature:
            feature.append( "PREV_"+str(index)+"_"+each_feature )

    ## Appends baseline additional features
    feature += [
        "isChangeSpeaker={}".format(isChangeSpeaker),
        "isFirstUtterance={}".format(isFirstUtterance),
        "SubSequenceNumber={}".format(speaker_sentence_counter),
        "DialogNumber={}".format(dialog_counter)
    ]

    ## Adds advanced features
    sub_feature = advanced_features(dialog.text)
    feature += sub_feature

    ## Adds the posTags and Tokens for he current utterance
    if(dialog.pos):
        for index, posTag in enumerate(dialog.pos):
            dialog_pos = posTag.pos
            feature +=  [
                "TOKEN_" + posTag.token,
                "POS_" + dialog_pos
            ]
            sub_feature += [
                "POS_" + dialog_pos
            ]
        pos_counter[dialog_pos] += 1

    ## Uses text as token if the dialog.pos field is empty
    else:
        feature += [
            "TOKEN_" + dialog.text,
            "POS_TEXT"
        ]

    ## Adds count of each posTag as a feature to the current utterance
    for pos in pos_counter:
        feature.append ( "CountOf:"+pos+"="+str(pos_counter[pos]) )

    return feature, sub_feature

def extract_features_and_labels(input_folder):

    ## Fetch the entire dataset (dialogue_set) using read_tool
    dialogue_set = read_tool.get_data(input_folder)

    ## initialize overall feature set for the training dataset
    dialogue_set_features = []
    dialogue_set_labels = []

    ## Set n value for ngram model
    ngram_value = 4

    ## For each conversation (dialogue) in the training dataset
    for dialogue in dialogue_set:

        ## Initialize features for each dialogue
        dialogue_features = []
        dialogue_labels = []
        previous_speaker = None
        current_speaker = None

        ## Initialise a deque to remember features for (n-1) previous utterances
        ngram_queue = deque( maxlen=(ngram_value-1) )

        ## For each utterance in a dialogue
        for index, utterance in enumerate(dialogue):

            ## Fetch tag, current speaker
            act_tag = utterance.act_tag if utterance.act_tag else "DEFAULT_TAG"
            current_speaker = utterance.speaker

            ## Evaluate if speaker is changed or not
            if current_speaker == previous_speaker:
                isSpeakerChange = False
            else:
                isSpeakerChange = True

            ## Evaluate features for the utterance
            feature, advanced_feature = generate_feature(utterance, isSpeakerChange, index==0, ngram_queue)

            ## Assign current speaker as the previous one before moving to next utterance
            previous_speaker = current_speaker

            ## Add the sub_features to deque so that next features can use this as previous features
            ngram_queue.appendleft(advanced_feature)

            ## Append utterance features, labels to the dialog
            dialogue_features += [feature]
            dialogue_labels += [act_tag]

        ## Merge features, labels for the entire dataset
        dialogue_set_features += [dialogue_features]
        dialogue_set_labels += [dialogue_labels]

    return dialogue_set_features, dialogue_set_labels

if __name__ == "__main__":

    ## Read input parameters from command line
    INPUT_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    ## Extract features and labels from training data
    features, labels = extract_features_and_labels(INPUT_DIR)
    
    ## Initialize a pycrf Model file
    Model = pycrf.Trainer(verbose=False)

    ## Generate and Train Model
    for xSequence, ySequence in zip(features ,labels):
        Model.append(xSequence, ySequence)

    ## Set parameters for training
    Model.set_params({
        'c1': 1.0,  # L1 penalty
        'c2': 1e-3,  # L2 penalty
        'max_iterations': 50
    })

    ## Start training
    Model.train('advanced_model.crfsuite')

    ## Tag the testing data
    Tagger = pycrf.Tagger()
    Tagger.open('advanced_model.crfsuite')

    ## Extract features for the Testing folder
    features, labels = extract_features_and_labels(TEST_DIR)

    ## Fetch all testing files for tagging
    output = open(OUTPUT_FILE, "w+")
    
    #correct, total = 0, 0
    
    ## For every test file, append label to the specified output file
    for file_number in range(len(features)):
        for utterance_index, label in enumerate( Tagger.tag(features[file_number]) ):
            
            #To find accuracy
            '''           
            if label == labels[file_number][utterance_index]:
                correct += 1
            total += 1
            '''
            
            label += "\n"
            output.writelines( label )
        output.writelines("\n")

    output.close()
    
    #print("Advanced Accuracy : ",correct/total)