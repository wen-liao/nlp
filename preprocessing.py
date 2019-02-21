import numpy as np

##label set
BMES_set = ["B-GPE","M-GPE","E-GPE","S-GPE","B-PER","M-PER","E-PER","S-PER",
              "B-LOC","M-LOC","E-LOC","S-LOC","B-ORG","M-ORG","E-ORG","S-ORG","O"]
entag_set = ["GPE","PER","LOC","ORG","O"]#entity tag
BMES2ix = dict()
for l in BMES_set:
    BMES2ix[l] = len(BMES2ix)
entag2ix = dict()
for l in entag_set:
    entag2ix[l] = len(entag2ix)

##file paths
TRAIN_W_PATH = r".\dataset\train.word.bmes"
TEST_W_PATH = r".\dataset\test.word.bmes"
DEV_W_PATH = r".\dataset\dev.word.bmes"
TRAIN_C_PATH = r".\dataset\train.char.bmes"
TEST_C_PATH = r".\dataset\test.char.bmes"
DEV_C_PATH = r".\dataset\dev.char.bmes"


#preprocessing
#Currently we only support BMES notations.
def preprocessing(PATH):
    sents = []
    label_seqs = []
    sent = []
    label_seq = []
    file = open(PATH,encoding = 'utf-8')
    for line in file:
        if len(line.strip()) == 0:
            sents.append(sent)
            label_seqs.append(label_seq)
            sent = []
            label_seq = []
        if len(line) > 3:
            char,label = line.strip('\n\ufeff').split(" ")
            sent.append(char)
            label_seq.append(label)
    file.close()
    return sents, label_seqs


##evaluation
def evaluate(pred_label_seqs, golden_label_seqs,notation = 'BMES'):
    #accuracy
    correct_labels = 0
    total_labels = 0
    SENT_NUM = len(pred_label_seqs)
    for i in range(SENT_NUM):
        total_labels += len(pred_label_seqs[i])
        for j in range(len(pred_label_seqs[i])):
            if pred_label_seqs[i][j] == golden_label_seqs[i][j]:
                correct_labels += 1
    if total_labels == 0:
        accuracy  = -1
    else:
        accuracy = correct_labels / total_labels
    
    pred_label_num = 0
    golden_labels_num = 0
    intersection_num = 0
    #precision, recall, f_measure
    if notation == "BMES":
        for i in range(SENT_NUM):
            preds = get_BMES_entities(pred_label_seqs[i])
            goldens = get_BMES_entities(golden_label_seqs[i])
            pred_label_num += len(preds)
            golden_labels_num += len(goldens)
            intersection_num += len(set(preds)&set(goldens))

    if pred_label_num == 0:
        precision = -1
    else:
        precision = intersection_num / pred_label_num
    if golden_labels_num == 0:
        recall = -1
    else:
        recall = intersection_num / golden_labels_num
    if (precision == -1) or (recall == -1) or intersection_num == 0:
        f_measure = -1
    else:
        try:
            f_measure = 2*precision*recall / (precision + recall)
        except:
            print(precision,recall)
            print(pred_label_num,golden_labels_num,intersection_num)
    """
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall Rate: ", recall)
    print("F_measure: ", f_measure)
    """
    return accuracy, precision, recall, f_measure


def get_BMES_entities(seq,notation = "BMES"):
    if notation == "BMES":
        entities = []
        begin = -1
        entity = ""
        for i in range(len(seq)):
            if seq[i].startswith("S-"):
                entities.append(seq[i][2:] + "[" + str(i) + "," + str(i))
            elif seq[i].startswith("B-"):
                begin, entity = i,seq[i][2:]
            elif seq[i].startswith("M-"):
                if begin >= 0 and entity != seq[i][2:]:
                    begin = -1
            elif seq[i].startswith("E-"):
                if begin >= 0 and entity == seq[i][2:]:
                    entities.append(entity + "[" + str(begin) + "," + str(i))
                begin = -1
        return entities