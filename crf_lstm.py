from BiLSTM_CRF import *
from preprocessing import *


training_data = preprocessing(TRAIN_W_PATH)
test_data = preprocessing(TEST_W_PATH)

word_to_ix = dict()
for sentence in training_data[0]:
    for char in sentence:
        if char not in word_to_ix:
            word_to_ix[char] = len(word_to_ix)
word_to_ix[UNSEEN_WORD] = len(word_to_ix)
ix_to_word = {word_to_ix[key]:key for key in word_to_ix}
VOC_SIZE = len(word_to_ix)

tag_to_ix = BMES2ix
tag_to_ix[START_TAG] = len(tag_to_ix)
tag_to_ix[STOP_TAG] = len(tag_to_ix)
ix_to_tag = {tag_to_ix[key]:key for key in tag_to_ix}
BMES_set += [START_TAG, STOP_TAG]

training_data = [(prepare_seq(sent,word_to_ix),prepare_seq(tags,tag_to_ix)) for sent,tags in zip(training_data[0],training_data[1])]
test_data = [(prepare_seq(sent,word_to_ix),prepare_seq(tags,tag_to_ix)) for sent,tags in zip(test_data[0],test_data[1])]


for EMBEDDING_DIM in [50,70,90]:
    for HIDDEN_DIM in [50,60,80]:

        NOISE_RATE = 0.01

        model = BiLSTM_CRF(VOC_SIZE, EMBEDDING_DIM, HIDDEN_DIM, tag_to_ix)
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Check predictions before training
        with torch.no_grad():
            precheck_sent = training_data[0][0]
            precheck_tags = training_data[0][1]
            print(precheck_tags)
            print(model(precheck_sent))
        print("Test Successfully.")

        file = open("wordlstm_crf.txt",mode="a+")
        file.write("Embed Dim: %d  Hidden Dim: %d  Noise Rate: %f\n"%(EMBEDDING_DIM,HIDDEN_DIM,NOISE_RATE))
        file.close()

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(15): # again, normally you would NOT do 300 epochs, it is toy data
            print(epoch)
            import time
            t1 = time.time()
            loss_t = 0
            for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()
                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.

                # Step 3. Run our forward pass.
                loss = model.loss(sentence, tags)
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                loss_t += loss.item()
                optimizer.step()
            t2 = time.time()
            file = open("wordlstm_crf.txt", mode="a+")
            file.write("epoch: %d  "%(epoch))
            file.write("train time: %f s  "%((t2-t1)/60))
            file.write("total loss: %f\n"%(loss_t))
            if epoch%1 == 0:
                with torch.no_grad():
                    true, pred = [], []
                    for sentence, tags in test_data:
                        true.append([ix_to_tag[tag.item()] for tag in tags])
                        pred.append([ix_to_tag[ix] for ix in model(sentence)])
                file.write("Accuracy: %f  Precision: %f  Recall Rate: %f  F_measure: %f\n"%evaluate(pred,true))
            file.close()
            torch.save(model,".\\WordBiLSTM_CRF_"+str(EMBEDDING_DIM)+ "_" + str(HIDDEN_DIM) + "_0.01")