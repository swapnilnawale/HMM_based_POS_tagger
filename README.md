HMM_based_POS_tagger
====================

HMM Based POS tagger using Viterbi's algorithm in Python

Problem 
 Description       : This program implements part of speech (POS) tagging for 
                     English sentences using hidden markov models.
                     Part of speech tagging refers to the process of finding
                     part of speech for the words in a English sentence. 
                     The hidden markov models (HMM) are used for this process.
                     HMM are the models which are defined by a set of states
                     and each of these states are associated with some other 
                     hidden properties of the state. The set of states here
                     for POS tagging are words from the sentences while the
                     hidden states are the POS tag for that state.
                     HMM for this POS tagging are represented by two kinds
                     of Probabilities:
                     a) Tag transition Probabilities : The probability that a
                     tag is followed by another tag.
                     b) Word observation likelihood: The probability that a
                     a word is tagged with a particular tag.
                     
                     These Probabilities are decided from an existing set
                     of English sentences (training set) and are then applied
                     to find the tags from another set of English sentences
                     (test set). To find the tag from HMM, viterbi's algorithm
                     is used in this program. The viterbi's algorithm used 
                     here is taken from the Section 5.5 of Jurafsky-Martin 
                     text "Speech and Language Processing".
                      
                     
 Usage             : This program takes following inputs:
                     1) -tr = the name of training file used for POS tagging. 
                     2) -ts = the name of test file which needs to be tagged.
                     3) -tk = the name of gold std file/ manually tagged file.
                     e.g. to run this program for a training file 
                     "pos-train.txt", test file "pos-test.txt" and gold std,
                      file "pos-test-key.txt" use 
                      following command

 python pos_tagging.py -tr pos-train.txt -ts pos-test.txt -tk pos-test-key.txt
                     
                     Please note that sequence of the inputs SHOULD be same as
                     shown above. i.e. -tr <training file name> 
                     -ts <test file name> 
                     
                     This program creates an output file with name 
                     "tagging-output", which contains the tagged words from 
                     test set. It also creates a csv file containing confusion
                     matrix of tagging, which denotes inaccuracies happened
                     in the tagger i.e. percentage of times a tag is 
                     incorrectly assigned to a word instead of other tag.
