import hw2_corpus_tool as read_tool
import pycrfsuite as pycrf
import sys

'''
Feature = 
    [
        isSpeakerChanged = True/False,
        isUtteranceStart = True/False,
        Token, Pos_Tag,
        Token, Pos_Tag,
        ...
    ]
'''
def generate_feature (utterance, is_speaker_change, is_dialogue_start):

    feature = []

    if is_dialogue_start:
        feature += ["DialogueStart=True", "SpeakerChanged=False"]
    else:
        feature += ["SpeakerChanged=" + str(is_speaker_change)]

    ## Handling empty POS
    if utterance.pos:
        for pos_tag in utterance.pos:
            feature += [ "TOKEN_"+pos_tag.token, "POS_"+pos_tag.pos ]
    else:
        feature += ["NO_WORD"]

    return feature

def extract_features_and_labels (input_folder):

    ## Fetch the entire dataset (dialogue_set) using read_tool
    dialogue_set = read_tool.get_data(input_folder)

    ## initialize overall feature set for the training dataset
    dialogue_set_features = []
    dialogue_set_labels = []

    ## For each conversation (dialogue) in the training dataset
    for dialogue in dialogue_set:

        ## Initialize features for each dialogue
        dialogue_features = []
        dialogue_labels = []
        previous_speaker = None
        current_speaker = None

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
            feature = generate_feature (utterance, isSpeakerChange, index==0)

            ## Assign current speaker as the previous one before moving to next utterance
            previous_speaker = current_speaker

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
        'max_iterations': 50,
        'feature.possible_transitions': True
    })

    Model.train('baseline_model.crfsuite')

    ## Tag the testing data
    Tagger = pycrf.Tagger()
    Tagger.open('baseline_model.crfsuite')

    ## Extract features for the Testing folder
    features, labels = extract_features_and_labels(TEST_DIR)

    ## Fetch all testing files for tagging
    output = open(OUTPUT_FILE, "w+")
    
    #correct, total= 0, 0

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
    
    #print("Baseline Accuracy : ",correct/total)