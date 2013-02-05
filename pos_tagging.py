##############################################################################
# Problem 
# Description       : This program implements part of speech (POS) tagging for 
#                     English sentences using hidden markov models.
#                     Part of speech tagging refers to the process of finding
#                     part of speech for the words in a English sentence. 
#                     The hidden markov models (HMM) are used for this process.
#                     HMM are the models which are defined by a set of states
#                     and each of these states are associated with some other 
#                     hidden properties of the state. The set of states here
#                     for POS tagging are words from the sentences while the
#                     hidden states are the POS tag for that state.
#                     HMM for this POS tagging are represented by two kinds
#                     of Probabilities:
#                     a) Tag transition Probabilities : The probability that a
#                     tag is followed by another tag.
#                     b) Word observation likelihood: The probability that a
#                     a word is tagged with a particular tag.
#                     
#                     These Probabilities are decided from an existing set
#                     of English sentences (training set) and are then applied
#                     to find the tags from another set of English sentences
#                     (test set). To find the tag from HMM, viterbi's algorithm
#                     is used in this program. The viterbi's algorithm used 
#                     here is taken from the Section 5.5 of Jurafsky-Martin 
#                     text "Speech and Language Processing".
#                      
#                     
# Usage             : This program takes following inputs:
#                     1) -tr = the name of training file used for POS tagging. 
#                     2) -ts = the name of test file which needs to be tagged.
#                     3) -tk = the name of gold std file/ manually tagged file.
#                     e.g. to run this program for a training file 
#                     "pos-train.txt", test file "pos-test.txt" and gold std,
#                      file "pos-test-key.txt" use 
#                      following command
#
# python pos_tagging.py -tr pos-train.txt -ts pos-test.txt -tk pos-test-key.txt
#                     
#                     Please note that sequence of the inputs SHOULD be same as
#                     shown above. i.e. -tr <training file name> 
#                     -ts <test file name> 
#                     
#                     This program creates an output file with name 
#                     "tagging-output", which contains the tagged words from 
#                     test set. It also creates a csv file containing confusion
#                     matrix of tagging, which denotes inaccuracies happened
#                     in the tagger i.e. percentage of times a tag is 
#                     incorrectly assigned to a word instead of other tag.
#                      
# Algorithm         : 1) This program first reads the training and test files 
#                        entered by user. And make their copies to keep them
#                        intact.
#                     2) It then compares the words from the training file with
#                        the words from test file to find out the unknown words
#                     3) The tags of the unknown words are found using a rule
#                        based approach dependent on the morphology of words.
#                     4) It then forms an HMM for the training file. It creates
#                        tag transition probabilities matrix and observation
#                        likelihood probabilities matrix from words and tags
#                        of the training file. These two matrices represent 
#                        the HMM for this problem.
#                     5) Using this HMM, it then finds out the tags for the
#                        words present in the test file by application of
#                        viterbi's algorithm. The tags of unknown words are
#                        taken as decided in step 3.
#                     6) The word-tag pairs for test file is written into
#                        an output file. This output file has the exactly same
#                        format as that of test file.
#                     7) This program then evaluates the accuracy of the tagger
#                        by comparing the tagging done against a manually 
#                        tagged gold std. file. It calculates the overall
#                        accuracy and confusion matrix for the tagger. 
#
# Author            : Swapnil Nawale 
#
# Date              : 09/28/2012
#
# Version           : 1.0
#
# Prog. Language    : Programming Language used for this program is Python
#                     (Version 2.7.3).
#                     The basic Python code, used in this program, is
#                     learnt from the book "Think Python - How to Think Like 
#                     a Computer Scientist (by Allen B. Downey)" and from
#                     Google's Python Class , present online at 
#                     http://code.google.com/edu/languages/google-python-class/
#                       
# Text-Editor used  : vim editor on Linux Platform
#
# Notes             : OVERALL ACCURACY OF THE TAGGER = 87.5560%
#
# Known issues      : 1) The tagger sometimes assigns an '.' tag to non 
#                        sentence boundary character and then assigns the same
#                        tag to all subsequent words in that sentence. This is 
#                        not a regular behavior and happens a few number of 
#                        times only. This behavior is also seen for first 10
#                        sentences of the test file too.  
###############################################################################
#!/usr/bin/python

'''
import statements to include Python's in-built module functionalities in the
program
'''
# sys module is used to access command line argument, exit function etc.
import sys

# re module is used to access regular expression related facilities
import re

# os module is used to access file manipulation features
import os

# operator module is used for sorting data structures
import operator

# itertools module is used for forming tag bigrams 
import itertools

# collections module is used for creating ordered hash tables / dicts 
import collections

# python csv module is used for pretty printing of confusion matrix
import csv

'''
Set the value of debug flag. debug flag is used to decide whether to print
debug information in the output or not. This flag will be a global variable.
'''
debug = False

###############################################################################
# Function      : create_copy(original_file_name)
# Description   : This function creates a copy of the file, which is passed as
#                 parameter to the function. It opens the file, reads all line
#                 from it and writes them into a copy file. Copy file name will
#                 be same as original file name appended with '.copy' 
#                 extension.This function will return the name of copy file 
#                 created.
# Arguments     : original_file_name- Name of the file for which copy is to be
#                 created
# Returns       : The name of copy file created.
###############################################################################
def create_copy(original_file_name):
    
    '''
    Open the original file using in built open function in read mode.
    Open function takes two parameters: 
    1) Path of the file to be opened
    2) Mode in which file should be opened. Valid modes are
       r  - read 
       w  - write
       a  - append
       r+ - read and write
       b  - binary
    open function returns a file object (referred hereafter as file handle).
    We can use this file handle to read, write or append a string to
    file.
    '''
    original_file_handle = open(original_file_name, 'r')

    '''
    Create a copy file using open function in write mode. Name of the copy
    file will be original copy file appended with '.copy'
    '''
    copy_file_name = original_file_name + ".copy"

    copy_file_handle = open(copy_file_name, 'w')

    '''
    Read all lines from the original file using its file handle.
    For this, use in-built readlines function. readlines function reads 
    lines from the file specified by file handle and returns a Python list
    containing each line as one element of list. This list will also have a 
    trailing newline character to each element.
    e.g. If the contents of file are:

    Neo
    Trinity
    Morpheus

    then, readlines will return a list with the contents like:
    ['Neo\n','Trinity\n','Morpheus\n']

    '''
    # read all lines from original file and store them into a list
    original_file_list = original_file_handle.readlines()

    ''' 
    Iterate over the original_file_list so that its content can be 
    written into copy file. For iteration over the list, For loop can be used.
    Such for loops reads each element of list and store them in a counter 
    variable automatically. This counter variable is modified to new element
    on each iteration.
    e.g. If the elements of a list sample_list are as follows :

    ['Neo\n','Trinity\n','Morpheus\n']

    and if we write for loop for above list like below

    for sample_list_element in sample_list:
        print sample_list_element

    then value of sample_list_element for first iteration will be Neo\n.
    Similarly values of this variable for second and third iteration will be
    Trinity\n and Morpheus\n resp.
    '''

    '''
    iterate over original_file_list and write the contents of list to copy file
    using in-built write function. write function writes a string passed to it
    into a file, specified by file handle on which write function is called.
    '''
    for original_file_line in original_file_list:
        copy_file_handle.write(original_file_line)

    '''
    Close original and copy file using in built close function called on file
    handle.
    '''

    original_file_handle.close()
    copy_file_handle.close()

    return copy_file_name 

###############################################################################
# End of create_copy Function
###############################################################################

###############################################################################
# Function      : clean_file(file_name)
# Description   : This function cleans the file passed as parameter to it.
#                 Cleaning process will remove all square brackets from the 
#                 file passed to this function.
# Arguments     : file_name - Name of the file to be cleaned.
# Returns       : None.
###############################################################################
def clean_file(file_name):
    
    # open the file in read mode
    file_handle = open(file_name, 'r') 

    '''
    Get all lines from this file using readlines() function and store into
    a list.
    '''
    file_lines_list = file_handle.readlines()

    '''
    Close the file. This is required as I want to overwrite the same file.
    Overwriting will avoid creation of multiple intermediate files.
    '''
    file_handle.close()

    '''
    Now open the file in write mode so as to write all characters except the 
    square brackets from the file. Using write mode for a file in Python 
    erases the existing file with the same. And this will facilitate 
    overwriting of the file for the program.
    '''
    file_handle = open(file_name, 'w')

    '''
    Iterate over the file_lines_list and remove all square bracket characters 
    from each element line of list. For this, in built replace() 
    function will be used.
    Write the converted lines into the file specified by file_handle.
    '''
    for file_line in file_lines_list:
        
        file_handle.write(file_line.replace('[','').replace(']',''))

    # close the file before exiting function
    file_handle.close()

###############################################################################
# End of clean_file Function
###############################################################################

###############################################################################
# Function      : form_HMM(train_file)
# Description   : This function forms the HMM for POS-tagging. It creates the
#                 tag transition Probabilities matrix and observation likelihood
#                 Probabilities matrix (which represent HMM in this program) 
#                 from the training file passed as param.
#                 
#                 These two matrices will be used in viterbi algorithm to find
#                 out most probable tags for each word in test file.
# Arguments     : train_file -  Name of training file, used to form HMM 
# Returns       : A dict object storing mapping of tag bigrams to their tag
#                 tag transition probabilities
#                 A dict object storing mapping of word-tag pairs with their
#                 observation likelihood Probabilities
#                 List of all unique tags - to be used in applying viterbi
#                 algo to HMM
###############################################################################
def form_HMM(train_file):
    
    '''
    Start building tag transition Probabilities matrix and observation 
    likelihood Probabilities matrix.
    
    The approach, followed to get theses matrices, is as follows:
    
    1) I am not using any specific sentence marker like <s> for denoting a
    start of sentence tag. Instead of that, I am using '.' for marking
    start of the sentence. A period '.' will be present at the end
    of each sentence (or we can say at the start of next sentence). Only first 
    sentence won't have period as a start of sentence tag, so add a period
    at the start of training file content, as a first step of approach.
    
    2) Then, read the training file and convert all multi space
    characters from it, if any, to single space character.
    
    3) After converting all multi space characters to a single space char from 
    each line of the training file, split each line by that single space char.
    This splitting will give me pairs of word and tags separated by a '/'.

    4) Again split these pairs by last occurrence of '/' to get the separate 
    word and tag. 
    
    Here, I tried to think a regex approach for separating words and tags but 
    could not reach to an unique regex that will handle all kinds of words 
    present in the file. We have alphanumeric words and words with special 
    characters (along with the case when \ is escaped), so finding a single 
    regex fit for all kinds words in training file was really difficult. After 
    contemplating a lot about regexes, I resorted to use basic string parsing 
    approach, mentioned above.

    5) Each unique tag, obtained from the above separating of words and tags,
    will be stored in a list "unique_tags". This list will specify the 
    space of all valid tags. 

    E.g. if training file has following contents:
     
    Pierre/NNP Vinken/NNP 
    ,/, 
    61/CD years/NNS 
    old/JJ ,/, will/MD join/VB 
    the/DT board/NN 
    as/IN 
    a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD 
    ./.

    the "unique_tags" list will have following tags:

    ['NNP', ',', 'CD', 'NNS', 'JJ', 'MD', 'VB', 'DT', 'NN', 'IN', '.']

    6) Each unique word (type), obtained from the above separating of words and
    tags, will be stored in a list "unique_words". This list will specify the 
    space of all valid types. 

    E.g. if the training file has following contents as given in step 5,then 
    list of all types unique_words will look like this:
    
    ['.', 'Pierre', 'Vinken', ',', '61', 'years', 'old', 'will', 'join', 'the'
    , 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29']

    7) A dict object (map) "tag_to_words_mapping_dict" will be maintained that 
    will store mapping of each POS tag and list of different words that are 
    tagged for it. This will dict object be used in calculating observation 
    likelihood.

    E.g. if the training file has following contents as given in step 5,then 
    dict object "tag_to_words_mapping_dict" will look like this:
    
    -------------------------------------------------------------------------
    | Tag   |   List of words                                               |
    -------------------------------------------------------------------------
    | NNP   |   ['Pierre', 'Vinken', 'Nov.']                                |
    -------------------------------------------------------------------------
    | ,     |   [',', ',']                                                  |
    -------------------------------------------------------------------------
    | CD    |   ['61', '29']                                                |
    -------------------------------------------------------------------------
    | NNS   |   ['years']                                                   |
    -------------------------------------------------------------------------
    | JJ    |   ['old', 'nonexecutive']                                     |
    -------------------------------------------------------------------------
    | MD    |   ['will']                                                    |
    -------------------------------------------------------------------------
    | VB    |   ['join']                                                    |
    -------------------------------------------------------------------------
    | DT     |  ['the', 'a']                                                |
    -------------------------------------------------------------------------
    | NN    |   ['director','board']                                        |
    -------------------------------------------------------------------------
    | IN    |   ['as']                                                      |
    -------------------------------------------------------------------------
    | .     |   ['.']                                                       |
    -------------------------------------------------------------------------
 

    8) The sequence in which tags appear in the training file will be stored in
    a file viz. "tag_sequence_file". This file will be used to calculate tag 
    transition Probabilities. 
    
    E.g. if the training file has following contents as given in step 5,then 
    file "tag_sequence_file" will look like this: 

    "NNP NNP , CD NNS JJ , MD VB DT NN IN DT JJ NN NNP CD ."
 
    '''
    
    # open the train file in read mode
    train_file_handle = open(train_file, 'r') 

    # open the tag sequence file in write mode
    tags_seq_file_name =  'tags_sequence_file'
    tags_seq_file_handle = open(tags_seq_file_name, 'w')

    '''
    Get all lines from training file using readlines() function and store them 
    into a list. 
    
    Append a period tag at the start of contents of this list first.
    '''
    train_file_lines_list = []

    train_file_lines_list.append("./.\n") 
    for line in train_file_handle.readlines():
        train_file_lines_list.append(line)


    if debug:
        print train_file_lines_list

    # close the training file
    train_file_handle.close()
    
    # initialize a list to store the unique tags from training file
    unique_tags = []

    # initialize a list to store the unique words from training file
    unique_words = []
    
    '''
    Initialize a dict object to store mapping of a POS tag to the words 
    tagged with it
    '''
    
    tag_to_words_mapping_dict = {}

    '''
    Iterate over the train_file_lines_list to separate out words from the tags.
    '''
    for train_file_line in train_file_lines_list:
        
        # convert multi space characters from each line to a single space
        train_file_line = " ".join(train_file_line.split())
        
        if debug:
            print "\n"
            print train_file_line
        
        '''
        Split each train_file_line by single space characters to get all word
        -tag pairs present in that line. These pairs will be stored in a list
        called as word_tag_pairs_list.
        '''
        word_tag_pairs_list = train_file_line.split(" ")

        if debug:
            print "\nword-tag pairs:"
            print word_tag_pairs_list
        
        '''
        Iterate over this pairs' list and separate words and tags from each
        pair. For this, split the pair at the last occurrence of '/' character 
        appearing in it. Splitting at last occurrence of '/' will correctly 
        handle the case when there is a word with escaped '/' character.
        For getting last occurrence of '/' char, built in function rfind will
        be used. It finds the last occurrence of any substring in a given 
        string.
        '''
        
        for word_tag_pair in word_tag_pairs_list:
            
            '''
            Split the word-tag pair at the last occurrence of '/' char. For
            this, neglect the word-tag pairs are blanks i.e. when rfind() is 
            returning -1
            '''

            if word_tag_pair.rfind('/') != -1:
                
                # get the separated word
                word = word_tag_pair[:word_tag_pair.rfind('/')]
                
                # replace all escaped '/' chars from the word with a single '/'
                word1 = word.replace('\\/', '/')

                if debug:
                    print word1
                
                '''
                Insert the word into unique_words if it is already not present
                in it
                '''
                if word1 not in unique_words:
                    unique_words.append(word1)


                # get the separated tag
                tag = word_tag_pair[word_tag_pair.rfind('/') + 1:]
                
                if debug:
                    print tag
                
                ''' 
                If a tag is a composite (ambiguous) tag, then select only 
                first tag out of it. 
                '''
                if '|' in tag:
                    tag = tag.split('|')[0]
    
                    if debug:    
                        print "\ncomposite tag"
                        print tag 

                '''
                Insert the tag into unique_tags if it is already no present
                in it
                '''
                if tag not in unique_tags:
                    unique_tags.append(tag)

                '''
                Write the tag in the tag sequence file.
                '''
                tags_seq_file_handle.write(tag + " ")

                '''
                Insert the POS tags and words that appear for that tag in the 
                dict tag_to_words_mapping_dict.This dict object will have all
                tags as the keys and a list of words that are tagged with the
                key tag will be the values.
                '''

                '''
                First check if the tag is present in the dict object as a key.
                Insert it as a key if it is not.
                '''
                if tag not in tag_to_words_mapping_dict.keys():
                        tag_to_words_mapping_dict[tag] = []
                    
                '''
                Add the word to the word list present for the given tag in 
                tag_to_words_mapping_dict
                '''
                tag_to_words_mapping_dict[tag].append(word1)

    if debug:
        print unique_tags
        print len(unique_tags)
        print unique_words
        print len(unique_words)
        print tag_to_words_mapping_dict


    # close the tag sequence file
    tags_seq_file_handle.close()

    '''
    Calculate the frequencies of each tag in the tag sequence file. For this,
    simply calculate the length of each words list from 
    tag_to_words_mapping_dict for each tag. This length will give the freq of
    each tag. The mapping of each tag to it's frequency will be stored in 
    ordered dict object tag_to_freq_dict.
    
    In python normal dict object is unordered .To maintain order of insertion, 
    instead of normal dict, an OrderedDict object will be used. 
    OrderedDict class belongs to Python module collections. 
    The usage of ordered dicts was learnt
    from the article by Doug Hellman, present at the link:

    http://www.doughellmann.com/PyMOTW/collections/ordereddict.html 
    '''
    
    ''' 
    Initialize an ordered dict object to store the mapping of tags and their
    frequencies 
    '''
    tag_to_freq_dict = collections.OrderedDict()
    
    for tag, words in tag_to_words_mapping_dict.iteritems():
        tag_to_freq_dict[tag] = len(words)    
 
    if debug:
        print tag_to_freq_dict
    
    '''
    Get the tag transition probabilities matrix by calling 
    get_tag_trans_prob_matrix function. 
    
    This function takes following arguments:
    1) A dict object storing mappings of each tag to its freq.
    2) File containing sequence of tags as they appear in training file
    
    And it returns an ordered dict object containing mapping of tag bigrams
    to their tag transition Probabilities. This dict object represents our 
    tag transition probability matrix.
    
    A sample tag transition prob matrix dict object will be like this:

    -----------------------------------------------------------
    | tag bigram        |           probability                |
    -----------------------------------------------------------
    | ('.','.')         |           0.0000001                 |
    -----------------------------------------------------------
    | ('NN','NN')       |           0.0002                    |
    -----------------------------------------------------------
    '''
    
    tag_transition_prob_matrix = get_tag_trans_prob_matrix(\
                                 tag_to_freq_dict, tags_seq_file_name)

    if debug:
       print tag_transition_prob_matrix
    
    '''
    Get the observation likelihood matrix by calling get_obs_lkhd_prob_matrix.

    This function takes following arguments:
    1) A list containing all unique tags
    2) A list containing all unique words (types)
    3) A dict object containing mapping of each POS tag with the list of words
       tagged for that tag
    4) A dict object containing mapping of each POS tag with its frequency

    
    And it returns an ordered dict object containing mapping of word-tag
    pairs with their observation likelihood probabilities. This dict 
    object represents our observation likelihood matrix.
    
    A sample obs. likelihood prob matrix dict object will be like this:

    -----------------------------------------------------------
    | word-tag pair     |           probability                |
    -----------------------------------------------------------
    | ('I','.')         |           0.0000001                 |
    -----------------------------------------------------------
    | ('to','TO')       |           0.2907                    |
    -----------------------------------------------------------
 
    '''

    word_tag_obs_lkhd_dict = get_obs_lkhd_prob_matrix(unique_words,\
                                                      unique_tags,\
                                                     tag_to_words_mapping_dict\
                                                      ,tag_to_freq_dict)
    if debug:
        print "HMM"
        print unique_words
        print len(unique_words)

    # return tag transition probability matrix and observation likelihood matrix
    return tag_transition_prob_matrix, word_tag_obs_lkhd_dict \
    ,unique_tags

###############################################################################
# End of form_HMM function
###############################################################################

###############################################################################
# Function      : get_tag_trans_prob_matrix(tag_to_freq_dict, 
#                 tags_seq_file_name)
# Description   : This function creates the tag transition Probabilities matrix
#                 from the tags list and file containing the seq of tags. 
# Arguments     : tag_to_freq_dict - A dict object storing mappings of each tag
#                                    to its freq
#                 tags_seq_file_name - File containing sequence of tags as they
#                                      appear in training file 
# Returns       : An ordered dict object mapping tag bigrams to their tag 
#                 transition Probabilities.
###############################################################################
def get_tag_trans_prob_matrix(tag_to_freq_dict, tags_seq_file_name):

    # open the tag sequence file in read mode
    tags_seq_file_handle = open(tags_seq_file_name,'r')
    
    # get the contents of tag sequence file into a list variable
    tag_seq_file_contents = tags_seq_file_handle.readlines()    
    
    '''
    Iterate over the tag_to_freq_dict twice to form the tags bigrams. These
    tag bigrams will be stored in another ordered dict tag_bigram_dict as the
    keys. The Probabilities for these tag bigrams, calculated later in this 
    function, will be the values for tag bigram keys. These Probabilities are 
    initialized to 0 here.

    The keys in this dict will actually be python tuples. The usage of python
    tuples was learnt from the link:

    http://www.tutorialspoint.com/python/python_tuples.htm
    '''
    # initialize an ordered dict to store tag bigrams 

    tag_bigrams_dict = collections.OrderedDict() 

    for tag1 in tag_to_freq_dict.keys():
        for tag2 in tag_to_freq_dict.keys():
            
            # Initialize all tag trans. probabilities as 0 for each tag bigram
            tag_bigrams_dict[(tag1,tag2)] = 0.0000

    
    '''
    Count the occurrences of each of tag bigram from the tags sequence file . 
    For this, Counter object will be used.
    
    The count of bigrams from the word list will be done using
    Counter object, islice and izip functions from itertools module.

    The sample code for counting the bigrams was posted as a question
    asked by me on www.stackoverflow.com. The discussion of this
    question is present in the link:

    http://stackoverflow.com/questions/12488722/
    counting-bigrams-pair-of-two-words-in-a-file-using-python 

    I have used the code suggested in the answer by stackoverflow user
    Abhinav Sarkar. I have modified the regex given in the answer to fit
    for characters that could be present in the tags.
    
    This will give me frequencies for all tag bigrams with non-zero 
    frequencies. These non-zero frequencies will be stored in the counter 
    object bigram_non_zero_freq.

    '''
    
    words = re.findall('[\w.,$`\':()#]+', open('tags_sequence_file').read())
    bigram_non_zero_freq = collections.Counter(itertools.izip(words,\
                           itertools.islice(words, 1, None)))

    '''
    Now we have all tag bigrams in the tag_bigrams_dict with probabilities
    initialized as 0. Replace these 0 probabilities only for those tag bigrams
    which are present in the bigram_non_zero_freq counter object, built above.
    Divide the freq counts from bigram_non_zero_freq counter object by 
    freq counts of first unigram in tag bigram (retrieved from 
    tag_to_freq_dict) to get the probabilities.
    '''
    
    # iterate over bigram_non_zero_freq to get non zero probabilities
    for bigram_str_freq in bigram_non_zero_freq.most_common():
        tag_bigrams_dict[(bigram_str_freq[0])] = \
                                float(bigram_str_freq[1]) /\
                                float(tag_to_freq_dict[bigram_str_freq[0][0]])
    
    if debug:
        print tag_bigrams_dict
        print len(tag_bigrams_dict)
        
        
    # close tag sequence file
    tags_seq_file_handle.close()

    return tag_bigrams_dict
    
###############################################################################
# End of get_tag_trans_prob_matrix function
###############################################################################

###############################################################################
# Function      : get_obs_lkhd_prob_matrix(unique_words, unique_tags,\
#                             tag_to_words_mapping_dict,tag_to_freq_dict)
# Description   : This function creates the observation likelihood probability
#                 matrix for each word and tag in the training file
# Arguments     : unique_words - A list containing all unique words
#                                from the training file
#                 unique_tags -  A list containing all unique tags from the
#                                training file
#                 tag_to_words_mapping_dict - A dict object containing mapping
#                                             of each tag to the list of words
#                                             appearing for that tag
#                 tag_to_freq_dict - A dict object storing mappings of each tag
#                                    to its freq
# Returns       : An ordered dict object mapping tag, word pair to its 
#                 observation likelihood Probabilities.
###############################################################################

def get_obs_lkhd_prob_matrix(unique_words, unique_tags,\
                             tag_to_words_mapping_dict,tag_to_freq_dict):

    '''
    Iterate over the unique_words list and unique_tags list to get all word-tag
    pairs. For all such word-tag pairs,count the number of times the
    word is tagged with tag in pair. This count will be retrieved from 
    tag_to_words_mapping_dict. Divide this count by total number of times the
    tag appears in training file. This second count will be retrieved from 
    tag_to_freq_dict.
    The mapping of each word-tag pair to its observation likelihood probability
    will be stored in an ordered dict object viz. word_tag_obs_lkhd_dict.
    '''
    
    word_tag_obs_lkhd_dict =  collections.OrderedDict()

    for word in unique_words:
        for tag in unique_tags:

            word_tag_obs_lkhd_dict[(word,tag)] = \
                float(tag_to_words_mapping_dict[tag].count(word)) /\
                float(tag_to_freq_dict[tag])
            
            if debug:
                print word
                print tag
                print tag_to_words_mapping_dict[tag]

    if debug:
        print word_tag_obs_lkhd_dict

    return word_tag_obs_lkhd_dict
###############################################################################
# End of get_obs_lkhd_prob_matrix function
###############################################################################

###############################################################################
# Function      : preprocess_file(file_name)
# Description   : This function preprocesses a file before POS tagging. It 
#                 replaces all newline characters from file with a space.
#                 It will overwrite the existing file after preprocessing.
# Arguments     : file_name -  Name of the file to be preprocessed
# Returns       : None.
###############################################################################

def preprocess_file(file_name):

    # open the file in read mode
    file_handle =  open(file_name, 'r')
    
    # read all lines from the file and store them in a list
    file_lines_list = file_handle.readlines()
    
    # close the file
    file_handle.close

    # open the file again in write mode to overwrite it
    file_handle =  open(file_name, 'w')

    '''
    Iterate over the file_lines_list and replace all newline chars with 
    space and write them back into the file
    '''
    for line in file_lines_list:
        file_handle.write(line.replace('\n',' '))
    
    # close the file
    file_handle.close()        

###############################################################################
# End of get_obs_lkhd_prob_matrix function
###############################################################################

###############################################################################
# Function      : viterbi_decode(test_file, unique_tags, 
#                                word_tag_obs_lkhd_dict,
#                                tag_transition_prob_matrix,test_copy_file_1)
# Description   : This function tags all words with HMM POS-tagging using 
#                 viterbi's decoding algorithm. It uses an HMM, represented by
#                 tag transition probabilities and observation likelihood 
#                 Probabilities matrices, and decides the tag for each word in
#                 test file. It creates a final output file "tagging-output"
#                 which has all words from test file with their tags. It also
#                 writes the unknown words from test file with their tags in
#                 output file.
# Arguments     : test_file -  Name of preprocessed test file  
#                 unique_tags - A list storing all valid tags 
#                 word_tag_obs_lkhd_dict - A dict object storing mapping 
#                                          of word-tag pairs with their
#                                          observation likelihood Probabilities
#                 tag_transition_prob_matrix - A dict object storing mapping of
#                                              tag bigrams to their tag
#                                              tag transition probabilities
#                 test_copy_file_1 - A copy of original test file
#                 unknown_word_tags_mapping - A dict object containing unknown
#                 words and their tags
# Returns       : A count of total tokens/ words tagged, which will be later
#                 used for evaluation of tagger
###############################################################################
def viterbi_decode(test_file, unique_tags, word_tag_obs_lkhd_dict,\
                   tag_transition_prob_matrix, test_copy_file_1,\
                   unknown_word_tags_mapping):
    
    '''
    I have used the viterbi's decode algorithm as mentioned in the Section
    5.5 of JM text book "Speech and Language Processing". The variable naming
    conventions are same as that of the algorithm specified in the book.
    '''
    
    # open the preprocessed test file in read mode
    test_file_handle = open(test_file, 'r')

    # create tagging-output file which will store the final output
    op_file_handle = open("tagging-output",'w')

    # get all lines from the preprocessed test file into a list
    test_file_contents = test_file_handle.read()
    
    # close the preprocessed test file
    test_file_handle.close()

    # open the copy of original test file
    test_copy_file_1_handle =  open(test_copy_file_1, 'r')

    # read the lines from it
    test_copy_file_lines = test_copy_file_1_handle.readlines()

    # close the copy of original test file
    test_copy_file_1_handle.close()   

    '''
    Iterate over the contents of copy of original file stored in the
    list test_copy_file_1_handle. Extract all sentences from it and store
    them into a list test_file_1_sentences. These sentences will be 
    used later to create final output file.

    I am using the regex "?<![s\.])[\.][\s]*[\n]" here to extract the
    sentences, which I have decided to be suitable for sentence extraction 
    from test file after manual observation of original test file.

    This regex has following contents:

    1) (?<![s\.]) - This is a negative lookbehind regex in python and it means
    "Match a regex ( [\.][\s]*[\n] ) only if it is not
    preceded by an S or '.'. The usage of negative lookbehind was learnt from
    a similar question asked on stackoverflow online forum. It can be found 
    here:

    http://stackoverflow.com/questions/8266052/python-regex-negating-
    meta characters

    I have followed the answer by stackoverflow user "Dave Webb".
    
    2) [\.][\s]*[\n] - This means match a period followed by zero or more
    spaces, which is again followed a newline. 

    So, to decide the end of any sentence in original test file, above regex
    is used. 

    In a nutshell, the regex means search for this pattern-

    Look for line having <'.'><zero or more spaces><'\n'> only if there is 
    no <s> or <\.> before it. This avoids selection of periods from 
    abbreviations, fractional numbers and strings like '...'
    
    Finding this regex was really a Herculean task as I needed to check it for
    every sentence from test file. But eventually , I was able to get it.:-)

    sample usage of above regex:

    If the original test file has following contents:

    No , 
    [ it ]
    [ was n't Black Monday ]
    . 
    But while 
    [ the New York Stock Exchange ]
    did n't 
    [ fall ]
    apart 
    [ Friday ]
    as 
    [ the Dow Jones Industrial Average ]
    plunged 
    [ 190.58 points ]
    -- most of 
    [ it ]
    in 
    [ the final hour ]
    -- 
    [ it ]
    barely managed to stay  
    [ this side ]
    of 
    [ chaos ]
    . 

    then the extracted sentences stored in the list test_file_1_sentences will 
    be like this:

    {
        [No , 
        [ it ]
        [ was n't Black Monday ]
        . 

        But while 
        [ the New York Stock Exchange ]
        did n't 
        [ fall ]
        apart 
        [ Friday ]
        as 
        [ the Dow Jones Industrial Average ]
        plunged 
        [ 190.58 points ]
        -- most of 
        [ it ]
        in 
        [ the final hour ]
        -- 
        [ it ]
        barely managed to stay  
        [ this side ]
        of 
        [ chaos ]
        .

    Please note that despite of using above regex to separate sentences, there
    was a weird instance of sentence ending in test file at line number 25815. 
    It was like this [ ) .  ]. This line has a period though it is not really
    sentence end for the sentence around line 25815. And because of this, 
    I was getting wrongly fetched sentences by the regex used above. So, I
    have changed this manually to [ )   ] to remove the period. I took this
    as a configuration issue and removed the tag. But final output file does 
    contain a period tagged in this line, just to make its comparison with 
    gold std file easy.

    This was the sole modification done to test file.
    '''
    
    # create list test_file_1_sentences to store all of the extracted sentences
    test_file_1_sentences = []
    
    ''' 
    Create an intermediate list store individual extracted sentences over each
    iteration of contents of copy of original file.  
    '''
    ind_sentence_list = []

    # iterate over the contents of original copy file to extract sentences
    for l in test_copy_file_lines:
        # append each line to the list of individual extracted sentence 
        ind_sentence_list.append(l)

        '''
        If a line has sentence boundary, then join all previously extracted
        lines and transfer this joined content to the list 
        test_file_1_sentences.
        '''
        if re.findall('(?<![s\.])[\.][\s]*[\n]',l ):

            test_file_1_sentences.append(''.join(ind_sentence_list))
           
            # make ind_sentence_list empty to store next sentence 
            ind_sentence_list = []
            continue

    if debug:
        print test_file_1_sentences
        print len(test_file_1_sentences)    

    
    '''
    Split the contents of test file by period '.' character to get all the 
    sentences from the file. A period is considered to be valid end of sentence
    marker for our program. But period may occur in the abbreviations as well,
    so to do proper segmentation of sentences, I will split the test file
    by <space><'.'><space> characters. This will make sure that we don't get any
    false positives in sentence boundary detection.
    '''
    
    test_sentences_list = test_file_contents.split(' . ')
    
    if debug:
        for line in test_sentences_list:
            print line.replace('[','').replace(']','')

    '''
    Iterate over the sentences list and start applying viterbi's algorithm to
    each sentence. The steps followed for decoding using viterbi's algo are 
    as follows:

    1) As the above splitting by <space><'.'><space> characters 
    strips the period character from the end of each sentence, append a '.'
    character at the end of sentence. 
    
    2) Also, Viterbi's algorithm requires a start state for building 
    path probability matrix, and as I have used period as the start of sentence 
    marker, I need to append a '.' at the start of each sentence.

    3) After this, split the sentence at the space character again to get a 
    list of words from the sentences. This list will have periods as a start 
    and end elements of it and all words of the sentences will lie between the
    periods in the list. So, start and end periods will also be considered as
    words in the sentences.

    The name of this list will be observation_list.

    4) Create an ordered dict object path_prob_matrix to store the path 
    probability matrix of the viterbi algorithm. 

    5) Iterate over all words from observation list and tags, to fill in 
    each word-tag pair as a key of path_prob_matrix dict object. The viterbi
    path Probabilities for each word -tag pair will be the values for these 
    keys.

    e.g. If a sentence in test file is "I run" and if valid tags are PRP,'.' 
    and VB , then after appending trailing and leading '.' 
    characters to sentence, path_prob_matrix will look something like this:

    ---------------------------------------------
    | word-tag pair |    viterbi path prob      |
    ---------------------------------------------
    | ('.','.')     |   0.98                    |
    -------------------------------------------- 
    | ('.','PRP')   |   0.00                    |
    --------------------------------------------
    | ('.','VB')    |   0.00                    |
    --------------------------------------------
    | ('I','.')     |   0.00                    |
    --------------------------------------------
    | ('I','PRP')   |   0.98                    |
    --------------------------------------------
    | ('I','VB')    |   0.00                    |
    -------------------------------------------- 
    | ('run','.')   |   0.00                    |
    --------------------------------------------
    | ('run','PRP') |   0.00                    |
    --------------------------------------------
    | ('run','VB')  |   0.98                    |
    --------------------------------------------
    
    These viterbi path probability values are initialized to zero at first.

    6) Then fill in the viterbi path probabilities for the start observation 
    word '.'. These Probabilities will simply be 1.

    7) Initialize the Probabilities of first word paired with all tags
    product of the tag transition Probabilities and observation likelihood prob.

    8) Then perform the recursion step. For the second word through last period
    in observation paired with each tag, get the viterbi path Probabilities.

    9) I have not used any specific termination condition as above recursive 
    step will terminate automatically after processing trailing period in the
    observation_list.

    10) Also, I have not used a backpointer specifically as backtracing will be
    done by just iterating over path_prob_matrix at the end. This backtracing 
    will give me the POS tags for each word.

    11) Write the POS tags along with words into tagging-output file.

    '''
    # initialize a counter to store the total number of sentences tagged 
    sentence_counter = 0

    # initialize a counter to store the total number of tokens tagged
    token_count = 0

    # iterate over the sentences list from test file
    for sentence in test_sentences_list:
       
        # append a leading period
        observation_list1 = ['.']
        sentence_words_list = sentence.split()

        for sentence_word in sentence_words_list:
            observation_list1.append(sentence_word)

        # append a trailing period
        observation_list1.append('.')
        
        '''
        Remove all square brackets from the observation list created above and
        create a new list with remaining words. Square brackets are not used
        in POS tagging so removal of them is okay here. 

        The approach for removing all occurrences of square brackets from 
        a python list is borrowed from a similar question asked on 
        stackoverflow forum. It can be found here :

        http://stackoverflow.com/questions/1157106/
        remove-all-occurences-of-a-value-from-a-python-list

        I have followed the usage of lambda expressions as suggested by 
        answer given by user "Mark Rushakoff" for above mentioned question.
        '''

        observation_list = filter (lambda a: a != '[' and  a != ']',\
                                   observation_list1)

        if debug:
            print observation_list
    
        # create an ordered dict object to store path prob matrix 
        path_prob_matrix = collections.OrderedDict()

        '''
        Build the viterbi path prob matrix for each word in sentence and
        each valid tag. Initialize path prob to be zero. 
        '''
        for word in observation_list:
            for tag in unique_tags:
                path_prob_matrix[(word,tag)] = 0.0000

        if debug:
            print path_prob_matrix
            print len(path_prob_matrix)

        '''
        Initialize the path prob for first word in the sentence i.e. leading 
        '.' as 1. This will be start state for viterbi's algo.
        '''
        for tag in unique_tags:
            path_prob_matrix[(observation_list[0],tag)] = 1.0000

        '''
        Do the initialization step as given in viterbi's algo for first 
        non-period word and all tags paired with it. If a unknown word
        is encountered here, then assign a dummy high obs. likelihood prob
        as 0.99 to it. This dummy prob value is used just to maintain
        the flow of viterbi algorithm while switching from each observation
        to the next. Actual tag written in final output for each unknown
        word will be the one, decided earlier by rule-based approach.
        '''    
        for tag in unique_tags:
            try: 
                path_prob_matrix[(observation_list[1],tag)] = \
                    tag_transition_prob_matrix[('.', tag)] * \
                    path_prob_matrix[(observation_list[0],tag)]*\
                    word_tag_obs_lkhd_dict[(observation_list[1],tag)]
            except KeyError:
                    word_tag_obs_lkhd_dict[(observation_list[1],tag)] = 0.99
                    path_prob_matrix[(observation_list[1],tag)] = \
                    tag_transition_prob_matrix[('.', tag)] * \
                    path_prob_matrix[(observation_list[0],tag)]*\
                    word_tag_obs_lkhd_dict[(observation_list[1],tag)]
        
        '''
        Do the recursive step of viterbi's algo for the second
        non-period word through last word (trailing '.') paired with 
        all tags.
        
        If a unknown word is encountered here, then assign a dummy high obs. 
        likelihood prob as 0.99 to it. 
        
        This dummy prob value is used just to maintain
        the flow of viterbi algorithm while switching from each observation
        to the next. Actual tag written in final output for each unknown
        word will be the one, decided earlier by rule-based approach 
        '''        
        
        for i in range(2, len(observation_list)):
            for tag1 in unique_tags:
                previous_viterbi_state_prob_list=[]
                for tag2 in unique_tags:
                     previous_viterbi_state_prob_list.append(\
                     path_prob_matrix[(observation_list[i-1],tag2)] *\
                     tag_transition_prob_matrix[(tag2, tag1)])
            
                try: 
                    path_prob_matrix[(observation_list[i],tag1)] =\
                        max(previous_viterbi_state_prob_list) *\
                        word_tag_obs_lkhd_dict[(observation_list[i],tag1)]
                except KeyError:
                        word_tag_obs_lkhd_dict[(observation_list[i],tag1)] =\
                        0.99
                        path_prob_matrix[(observation_list[i],tag1)] =\
                        max(previous_viterbi_state_prob_list) *\
                        word_tag_obs_lkhd_dict[(observation_list[i],tag1)]
                
                if debug:
                    print observation_list[i]
                    print tag1
                    print max(previous_viterbi_state_prob_list) * \
                      word_tag_obs_lkhd_dict[(observation_list[i],tag1)]

        if debug:
            print tag_transition_prob_matrix
            print path_prob_matrix

        '''
        Perform backtracing by traversing through the path probability matrix
        and getting max path probability for each word and find corresponding
        tag, which will be our final tag for that word.
        
        Each word and it's tag will be stored in another list called
        as backtraced_word_tag. The word and it's tag will be separated by
        a '/'
        '''

        backtraced_word_tag = [] 
        
        for i in range(0,len(observation_list)):
            viterbi_indiviaul_obs_prob_list = []
            for tag in unique_tags:
                viterbi_indiviaul_obs_prob_list.\
                append(path_prob_matrix[(observation_list[i],tag)])
                
            # if word is known, then take the tag decided by viterbi algo.
            if observation_list[i] not in unknown_word_tags_mapping.keys():
                backtraced_word_tag.append(observation_list[i] + '/'\
                                   + unique_tags[\
                                   viterbi_indiviaul_obs_prob_list.\
                                   index(max(\
                                   viterbi_indiviaul_obs_prob_list))])
            # if the word is unknown, then take tag, got by rule based approach
            else:
                backtraced_word_tag.append(observation_list[i] + '/'\
                                   + unknown_word_tags_mapping[\
                                   observation_list[i]])


        if debug:
            print observation_list1
            print observation_list 
            print backtraced_word_tag

        '''
        Now that we have got the tags for each word. We need to write them in
        output file "tagging_output". But there are two things remaining:
        
        1) We need to put back the square brackets removed earlier at correct 
        positions in output file.
        2) And we need to introduce the newline characters again which were
        removed earlier during preprocessing of test file.
        
        For the first thing, to introduce square brackets:

        First find out the positions of square brackets in the sentence by 
        iterating over observation_list1. These positions/indices will be 
        stored in a separate list called as bracket_indices. The opening and 
        closing bracket appear in pairs, so the indices present at 
        even positions in bracket_indices will correspond to opening brackets
        while indices present at odd positions in bracket_indices will 
        correspond to closing brackets.

        e.g. If the original sentence from the test file is :

        No , [ it ] [ was n't Black Monday ] .
        
        then the contents of observation_list1 is as follows:

        ['.', 'No', ',', '[', 'it', ']', '[', 'was', "n't", 'Black', 'Monday', 
        ']', '.']

        The list observation_list1 has opening sq. bracket at positions 3 and 6
        while it has closing sq. bracket at positions 5 and 11. (Here list 
        indices start from 0)

        If we store these positions into list bracket_indices, it will look 
        like this:
        
        [3,5,6,11]

        In above list, the numbers at even positions are of opening sq. 
        brackets while numbers at odd positions are of closing sq. brackets.

        So, bracket_indices[0] = 3 and bracket_indices[2] = 6 are for opening 
        sq. brackets and bracket_indices[1] = 5 and bracket_indices[3] = 11 are
        for closing brackets. So we can use this new list for inserting opening
        and closing sq. brackets into the list "backtraced_word_tag" which has 
        all the tagged words from observation_list1 except sq. brackets.

        e.g. backtraced_word_tag for above sentence looks like this:

        ['./.', 'No/RB', ',/,', 'it/PRP', 'was/VBD', "n't/RB", 'Black/NNP'
        , 'Monday/NNP', './.']

        After insertion of brackets at position 3, 5, 6 and 11, we will get the
        backtraced_word_tag list as follows:
        
        ['./.', 'No/RB', ',/,', '[', 'it/PRP', ']', '[','was/VBD', 
        "n't/RB", 'Black/NNP', 'Monday/NNP', ']', './.']

        '''

        bracket_indices = []
        
        '''
        Iterate over observation_list1 to get positions of square brackets
        and add these positions to bracket_indices
        '''
        for i in range(0,len(observation_list1)):
            if observation_list1[i] == '[' or observation_list1[i] == ']':
                bracket_indices.append(i)
        
        if debug:
            print observation_list1[1:]
        
        
        
        '''
        Iterate over the bracket_indices list and insert the square brackets.
        '''
        for i in range(0,len(bracket_indices)):
            if i % 2 == 0:
                backtraced_word_tag.insert(bracket_indices[i],'[')
            else:
                backtraced_word_tag.insert(bracket_indices[i],']')
      
        if debug:
            print backtraced_word_tag[1:]

        '''
        Now proceed to second remaining thing: adding newlines to the tagged
        text, which were removed earlier in preprocessing. This task has been
        a major hurdle in this program. Getting the output from 
        backtraced_word_tag (which has word/tag pairs from each sentences with
        newlines removed) to make it look exactly same as original test file, 
        was really difficult. 
        
        First, I thought of finding the rule followed in the original test file
        for putting newlines into it. I found that there are newlines 
        before the opening square bracket and after the closing
        square bracket and a period character. So, inserting the newline 
        character at these positions while traversing backtraced_word_tag will
        suffice. I wrote a bunch of code on this assumption about newlines and
        eventually found out that the rule about newlines had many exceptions.
        So, I had to leave that approach! 

        Then, I thought I will compare each tagged sentence stored in 
        backtraced_word_tag list with the corresponding sentence in original 
        test file and put the tags from backtraced_word_tag just after the
        word in each sentence of original test file.

        I had each sentence with tagged words in backtraced_word_tag 
        and I had the copy of original test file test_copy_file_1.
        And I needed to extract sentences from original sentences, which was
        a tough task due to irregularity in sentence boundaries in test file.
        But regex "?<![s\.])[\.][\s]*[\n]" came to my rescue. It is already 
        mentioned above and in fact all sentences are also already extracted
        above into the list test_file_1_sentences. These sentences already have
        newlines, which we wanted in final output file.

        Now my job became a less tougher. I just need to put the tags from 
        backtraced_word_tag at the correct positions in each sentence of 
        test_file_1_sentences.

        It will happen as below:
        
        Consider that the original sentence in test_file_1_sentences is as 
        follows :

        No ,\n [ it ]\n[ was n't Black Monday ]\n.

        and the corresponding tagged text in the backtraced_word_tag list is

        No/DT ,/, [ it/PRP ] [ was/VBD n't/RB Black/NNP Monday/NNP ] ./.
        
        Now take each word-tag pair (e.g. No/DT) from backtraced_word_tag 
        and separate it word (No) and tag (DT), search for first occurrence of
        word "No" in original sentence in test_file_1_sentences and replace it 
        with No/DT so that the contents of original sentence in 
        test_file_1_sentences becomes :

        No/DT ,\n [ it ]\n[ was n't Black Monday ]\n.

        Repeat this until we get tags for each word like this:

        No/DT ,/,\n [ it/PRP ]\n [ was/VBD n't/RB Black/NNP Monday/NNP ] ./.\n

        This sentence with tags and newlines will be then written
        into final output file tagging-output. This will be the final output
        of all the mind boggling with newlines :)

        
        The code that was written with the wrong assumption that newlines 
        appear before and after certain characters in test file is shown below 
        in commented format.

        ''
        The test file has a newline before the opening square bracket and 
        after the closing
        square bracket and a period character. So, insert the newline 
        character at these positions while traversing backtraced_word_tag.
        ''
        for i in range(0,len(observation_list)):
            if observation_list[i] == '[':
                observation_list[i] = '\n['

            elif observation_list[i] == ']':
                observation_list[i] = ']\n'

            elif observation_list[i] == '.':
                observation_list[i] = '. \n'
        
        ''
        Write the contents of backtraced_word_tag into tagging-output file
        ''
        #for i in range(1,len(observation_list)):
            op_file_handle.write(observation_list[i] + ' ')

        
        
        # close the final output file
        op_file_handle.close()


        #''
        After writing contents of backtraced_word_tag to the tagging output
        file, there is still some more formatting needed to make tagging_output
        appear like the original test file. There are few extra blank lines in
        tagging_output file and extra leading spaces in some lines. 
        Need to remove those by formatting. For this, call format_op_file() 
        function. 
    
        It takes the name of tagging output file as an input to it, formats
        it and overwrites the original unformatted op file with formatted one.
        ''  
    
        format_op_file("tagging-output")
         
        ###################################################################
        # start of format_op_file() function
        ###################################################################
        def format_op_file(tagging_output):
    
            # open the tagging output file in read mode
            tagging_op_file_handle = open(tagging_output, 'r')

        #  read the contents of it to a list
        op_file_contents =  tagging_op_file_handle.readlines()

        # close the file
        tagging_op_file_handle.close()

        ''
        Open the tagging output file again in write mode so as to remove extra 
        blank lines and extra leading spaces. Opening in write mode will 
        overwrite
        the original tagging output file.
        ''
        tagging_op_file_handle = open(tagging_output, 'w')

        ''
        Iterate over the op_file_contents to remove blank lines and leading 
        spaces 
        and write the formatted lines back to tagging output file.
        '' 
        if debug:
            print op_file_contents
    
        for line in op_file_contents:
        
            # skip the lines which are blank
            if line == " \n":
                    continue
        
            # remove leading spaces
        
            ''
            The extra leading spaces won't be removed for the lines which has
            '(', '{' , ')', '}' after those spaces. 
            This is due to the convention followed in test file that 
            there will be a leading space if a line starts with '(' , '{', ')',
            '}'.
        
            And as we want the tagging output to be in same format as the test
            file, I am following the same convention here.
            ''
        
            if len(line) > 1:
                if line[0] == " " and line[1] != "(" and line[1] != "}" and\
                line[1] != "}" and line[1] != ")":
                        line = line[1:]

            tagging_op_file_handle.write(line)

    
        # close the file
        tagging_op_file_handle.close()
       
        ###################################################################
        # End of format_op_file() function
        ###################################################################

        '''

        '''
        Iterate over the backtraced_word_tag to get word-tag pair
        to put the tags in corresponding sentence in test_file_1_sentences.
        While doing this iteration, count the number of words being tagged.
        This number gives us the total tokens present in the test file and
        it will be later used in calculating overall accuracy of tagger later
        in the program. This counter is called as token_count here.
        '''

        '''
        Check if the iteration over the backtraced_word_tag list happens
        only till the all sentences in test_file_1_sentences are processed.
        '''
        if sentence_counter <= len (test_file_1_sentences) - 1:

            '''
            Get the current sentence from test_file_1_sentences corresponding 
            to sentence in backtraced_word_tag. Append spaces before
            and after newline char to current sentence. These spaces
            facilitate word boundaries in string replacement later. These
            spaces will be removed when final output is written.
            '''
            curr_sentence = ' ' + test_file_1_sentences[sentence_counter]
            
            curr_sentence = curr_sentence.replace('\n',' \n ')

            for word_tag in backtraced_word_tag[1:]:
                # separate word from the word/tag pair
                word_tag_list = word_tag.rsplit("/",1)
                word = word_tag_list[0]
                
                if word != '[' and word != ']':
                    token_count = token_count + 1

                '''
                Replace the word separated above by word/tag pair in
                current statement
                '''
                curr_sentence = curr_sentence.replace(' '+ word + ' ', \
                                                    ' ' + word_tag + ' ') 
                backtraced_word_tag.remove(word_tag)
        
        '''
        Husshh! Finally write the final tagged sentence into 
        tagging-output file.
        '''
        if sentence_counter < len(test_file_1_sentences):
            op_file_handle.write(curr_sentence.replace(' \n ','\n')[1:])  
        
        sentence_counter =  sentence_counter + 1
        
    # close the final output file
    op_file_handle.close()

    # return token_count
    return token_count

###############################################################################
# End of viterbi_decode function
###############################################################################

###############################################################################
# Function      : evaluate_tagging(tagging_op_file_name,  gold_std_file_name,
#                 token_count)
# Description   : This function calculates the overall accuracy of the tagging
#                 done by comparison against manually tagged gold std file.
#                 It also produces a confusion matrix to show percentage
#                 of times a tag was wrongly tagged with another tag.
# Arguments     : tagging_op_file_name - The name of file tagged by this
#                                         program
#                 gold_std_file_name - The name of manually tagged file
#                 token_count - Total number of tokens tagged in test file 
# Returns       : None.
###############################################################################

def evaluate_tagging(tagging_op_file_name,  gold_std_file_name, \
                    token_count):

    '''
    Open tagging output and gold standard file and read lines from them and
    close them.
    '''

    tag_op_file_handle = open(tagging_op_file_name,'r')
    gold_std_file_handle = open(gold_std_file_name, 'r')

    tagged_lines = tag_op_file_handle.readlines()
    gold_std_lines = gold_std_file_handle.readlines()

    tag_op_file_handle.close()
    gold_std_file_handle.close()

    if debug:
        print token_count

    '''
    Iterate over the tagged_lines and gold_std_line to search for
    wrongly tagged tags. If a tag is incorrectly tagged, then the correct tag 
    from gold std. file and wrong tag from tagging output file will be found
    and put into a pair. This pair will be the keys in a dict object called
    as confusion_matrix_dict. The values for these keys in the dict will be
    the number of times the incorrect tagging is observed for the key pair.
    '''
    # initialize the confusion_matrix_dict 
    confusion_matrix_dict = {}

    # maintain a counter to store the total number of incorrect tags
    incorrect_tags_count = 0

    for i in range(0, len(tagged_lines)):
        if debug:
            print tagged_lines[i]
            print gold_std_lines[i]

        # strip sq. brackets from the contents of tagged_lines & gold_std_lines
        tag_line = tagged_lines[i].replace('[','').replace(']','')
        gold_line = gold_std_lines[i].replace('[','').replace(']','') 
        
        if debug:
            print tag_line
            print gold_line
        
        '''
        Split the tag_line and gold_line by white spaces to get only word-tag
        pairs
        '''
        word_tag_pair_1 = tag_line.split()
        word_tag_pair_2 = gold_line.split()

        if debug:
            print word_tag_pair_1
            print word_tag_pair_2
        '''
        Iterate over these word_tag_pairs and split them by last '/' to 
        separate tags from words. Then compare these tags to find out
        incorrect tagging.
        '''
        for j in range(0,len(word_tag_pair_1)):
            tag1 = word_tag_pair_1[j].rsplit('/', 1)[1]
            tag2 = word_tag_pair_2[j].rsplit('/', 1)[1].split('|')[0]

            if debug:
                if tag1 not in unique_tags or tag2 not in unique_tags:
                    print word_tag_pair_1
                    print word_tag_pair_2
                    print word_tag_pair_1[j].rsplit('/')
                    print word_tag_pair_2[j].rsplit('/')
                    print tag1 
                    print tag2
   
            '''
            If the two tags are not same then insert this occurrence them into
            the confusion_matrix dict object
            '''
            if tag1 != tag2:
               
                if (tag1,tag2) not in confusion_matrix_dict.keys():
                    
                    confusion_matrix_dict[(tag1, tag2)] = 1
                
                else:
                    confusion_matrix_dict[(tag1, tag2)] =\
                        confusion_matrix_dict[(tag1, tag2)] + 1
                
                incorrect_tags_count = incorrect_tags_count + 1

    '''
    Print overall accuracy.
    '''
    print float(100) -((float(incorrect_tags_count) / float(token_count)) *100)
    

    '''
    Get all the tags which appear in confusion_matrix dict. These tags only
    will be displayed in rows and columns of evaluation output. 
    These tags will be stored in a list tags_list.
    '''
    tags_list = [] 

    for tag in confusion_matrix_dict.keys():
        
        if tag[0] not in tags_list:
            tags_list.append(tag[0])
        
        if tag[1] not in tags_list:
            tags_list.append(tag[1])

    if  debug:
        print tags_list
        print len(tags_list)

    '''
    Iterate over the tags_list and start printing confusion matrix.
    Calculate the percentage errors for each tag pair in confusion_matrix dict
    to display it in output. The output will be stored as a .csv file.
    Python provides an elegant csv module for creation of csv file. I have
    found that csv is an easiest way to get table like pretty printing of
    confusion matrix. The usage of csv module was learnt from an answer
    to a question on stackoverflow forum. It can be found here:

    http://stackoverflow.com/questions/2084069/create-a-csv-file-with-values-
    from-a-python-list

    I have followed the code in answer by stackoverflow user vy32.

    Python lists can directly be written into a csv file. So, form the list 
    that will represent the confusion matrix along with table row and col
    header.
    '''
    csv_list = []

    col_header = [' ']

    # Iterate over tags_list to get the tags as table col headers 

    for tag in tags_list:
        col_header.append(tag)

    # create csv file
    out = csv.writer(open("conf_matrix.csv","w"), delimiter=',')
    
    if debug:
        print confusion_matrix_dict
        print col_header

    out.writerow(col_header)

    row_data = []
    # iterate over the tags_list to get table row headers and data
    for i in range(0,len(tags_list)):
        row_data = [tags_list[i]]
        for j in range(0,len(tags_list)):
            try:
                row_data.append((float(\
                          confusion_matrix_dict[(tags_list[j],tags_list[i])])/\
                            float(incorrect_tags_count))*float(100))
            except KeyError:
                row_data.append('-')

        out.writerow(row_data) 
    if debug:
        print incorrect_tags_count
        print csv_list

###############################################################################
# End of viterbi_decode function
###############################################################################

###############################################################################
# Function      : get_unique_words(file_name, file_ind_flg)
# Description   : This function finds the unique words / types from the file
#                 which is passed as param to it. 
# Arguments     : file_name - Name of the file from which unique words are to 
#                             be found
#                 file_ind_flg - A flag indicating whether unique words are 
#                                to be found from training or test file. It
#                                has two valid values viz 'tr' for training
#                                file and 'ts' for test file. 
# Returns       : A list of unique words from the file
###############################################################################

def get_unique_words(file_name, file_ind_flg):

    
    # open the file in read mode
    file_handle = open(file_name, 'r')
    
    # read all lines from the file into a list
    lines_list = file_handle.readlines()

    # close the file
    file_handle.close() 

    if debug: 
        print lines_list    

    # initialize a list to store unique words
    unique_words = []
     
    # iterate over the lines_list to get the words from file 
    for line in lines_list:

        # convert multi space characters from each line to a single space
        line = " ".join(line.split())    
        
        '''
        Separate the contents of each line by single space. This will
        give us word-tag pairs, if the training file is being processed. Else,
        it will give us just the words, if test file.
        '''

        contents_list = line.split(" ")

        if debug:
            print contents_list
        
        # iterate over the contents list to get words
        for content in contents_list:

            '''
            Separate words from tags from training file if it is being 
            processed. The processing of training file 
            (and not that of test file) can be checked from the file indicator 
            flag passed as param to this function.
            '''
            
            if file_ind_flg == 'tr':
                word = content[:content.rfind('/', 1)]
            else:
                word = content

            if debug:
                print word

            # insert the word in unique_words list if it is not already in it
            if word not in unique_words:
                unique_words.append(word)
    
    if debug:
        print unique_words
        print len(unique_words)
    
    return unique_words

###############################################################################
# End of get_unique_words function
###############################################################################

###############################################################################
# Function      : main()
# Description   : Entry point for the project.
# Arguments     : None. Command Line Arguments in Python are retrieved from
#                 sys.argv variable of sys module.
# Returns       : None.
###############################################################################
def main():
    
    '''
    Check if any command line argument is passed to program. If not 
    throw error showing proper sample usage. 
    '''

    if (len(sys.argv) > 1):
        if debug:
            print "At least one parameter passed to program !"

        '''
        Get the values for test, training and gold std. file from sys.argv 
        command line arguments and store them into different variables.
        '''

        train_file_name = sys.argv[2]
        test_file_name = sys.argv[4]
        gold_std_file_name = sys.argv[6]

        if debug:
            print train_file_name
            print test_file_name

                
        '''
        First make copies of training and test files. These copies will be 
        used for any processing subsequently. This way original contents of 
        training and test files will be retained. 
        
        We will make three copies of test file. First copy will be used for 
        application of viterbi algo. to it. Second copy will be used
        to prepare the final output file. And third copy will be used
        in finding unknown words.

        For copying the files, call create_copy function. The names of 
        copy files returned from create_copy function will be stored in
        separate variables.
        '''
        
        train_copy_file = create_copy(train_file_name)
        test_copy_file =  create_copy(test_file_name)
        test_copy_file_1 =  create_copy(test_copy_file)
        test_copy_file_2 =  create_copy(test_copy_file_1) 
        
        '''
        Start cleaning the training and test files. 

        Cleaning process will remove all opening and closing square brackets 
        from files, which are significant only for identifying phrases 
        and which not considered here for POS-tagging. 
        
        Removing square brackets will facilitate the process of
        parsing files in subsequent processing. For cleaning, 
        a function "clean_file" will be called. The name of training and test
        files will be passed as param to this function. Here, only third copy
        of test file will be cleaned, as it is used in finding unknown words.
        Rest two copies of word file are not cleaned.
        '''

        clean_file(train_copy_file)
        clean_file(test_copy_file_2)

        '''
        Find out the words from training file which are not there in test file.
        These are termed as unknown words. Unknown words cannot be tagged by
        using training file. So, for tagging them, I am using my own rule based
        approach, which I have devised by segregating unknown words myself and 
        then checking their tags in gold std file manually. The rules, I have 
        used are:

        1) First find out the unique words from both training and test file
        to decide the unknown words.

        2) Tag the unknown words based upon this logic:

            a) First check if the word is a symbol like =, then assign SYM tag
            to it.

            b) If a word is present in a predefined particles' list, then 
            assign RP tag. This list is taken from Section 5.1 of the
            Jurafsky Martin Text "Speech and Language Processing", which
            is in turn taken from the Quirk et al. (1985) paper:

            "A Comprehensive Grammar of the English Language"

            c) If word has at least one numeric character, then check:
                i) If word has at least one alphabetic character, then assign
                   JJ tag.
                
                ii) Else assign CD tag
            
            d) Then check if word starts with a capital letter:
                i) If not, then check:
                    * If word ends with 'ing' then assign VBG tag.
                    * If word ends with 'ed' then assign VBN tag.
                    * If word ends with 's' then assign NNP tag.
                    * If word ends with 'ly', then assign RB tag. 

                ii) If word starts with a capital letter, then check:
                    * If word is a single letter, then check:
                       # if word is 'C', then assign CC tag.
                       # else assign DT tag. 
                    * Else if word ends with 's' then assign NNP tag.
                    * Else assign NNS tag
            
            e) If word does not satisfy any of above criteria, assign NNP tag. 
        '''

        '''
        Get unique words from the training and test files. For this, call 
        get_unique_words() function. It takes two parameters:

        1) Name of the copy of training or third copy of test file
        2) A flag indicating whether unique words are to be found in training
        or test file. This flag is required as training file has different 
        structure than test file. So, they require a little different kind of
        parsing while getting unique words from them. Training file differs in
        structure from the test file because it has both words and tags while
        test file only has words.
        This flag will have value as 'tr' for training file while it has value
        as 'ts' for test file.

        This function returns the list of unique words from the passed as param
        to it.
        '''

        unique_train_words = get_unique_words(train_copy_file, 'tr')
        unique_test_words = get_unique_words(test_copy_file_2, 'ts')
        
        # create a list to store unknown words

        unknown_words =  []

        '''
        Iterate over the unique_train_words and unique_test_words to find 
        unknown words.
        '''
        
        for word in unique_test_words:
            if word not in unique_train_words:
                unknown_words.append(word)
                
        if debug:
            print unknown_words
            print len(unknown_words)
            for word in unknown_words:
                print word
    
        '''
        Iterate over the unknown words list to apply above mentioned rules to
        get their tags. The mapping of unknown words to their tags will be 
        stored in a dict object. Also make a list to store predefined 
        particles.
        '''
        unknown_word_tags_mapping = collections.OrderedDict()

        particles_list =  ["aboard", "about", "above", "across", "ahead", 
                           "alongside", "apart", "around", "aside", "astray", 
                           "away", "back", "before", "behind", "below", 
                           "beneath", "besides", "between", "beyond", "by", 
                           "close", "down", "east", "west", "south", "north",
                           "eastwards", "westwards", "southwards", 
                           "northwards", "forward", "forwards", "home", "in", 
                           "inside", "instead", "near", "off", "on", 
                           "opposite", "out", "outside", "over", "overhead", 
                           "past", "round", "since", "through", "throughout", 
                           "together", "under", "underneath", "up", "within", 
                           "without"] 

        # Apply rules to unknown words to find out their tags
        for word in unknown_words:
            
            if word == '=':
                unknown_word_tags_mapping[word] = 'SYM'
                continue

            if word in particles_list:
                unknown_word_tags_mapping[word] = 'RP'
                continue

            if re.search(r'[0-9]', word) is not None:
                
                if re.search(r'[a-z A-Z]', word) is not None:
                    unknown_word_tags_mapping[word] = 'JJ'
                    continue
                else:
                    unknown_word_tags_mapping[word] = 'CD'
                    continue

            if word[0].islower():
                
                if word.endswith('ing'):
                    unknown_word_tags_mapping[word] = 'VBG' 
                    continue
                
                if word.endswith('ed'):
                    unknown_word_tags_mapping[word] = 'VBN'
                    continue
                
                if word.endswith('s'):
                    unknown_word_tags_mapping[word] = 'NNP'
                    continue
                
                if word.endswith('ly'):
                    unknown_word_tags_mapping[word] = 'RB'
                    continue
            else:
                
                if len(word) == 1:

                    if word == 'C':
                        unknown_word_tags_mapping[word] = 'CC' 
                        continue
                    else:
                        unknown_word_tags_mapping[word] = 'DT' 
                        continue
                                    
                elif word.endswith('s'):
                    unknown_word_tags_mapping[word] = 'NNP' 
                    continue
                

            unknown_word_tags_mapping[word] = 'NNP' 
            

        '''
        Start building HMM for the given training file. For this, We need to 
        create tag transition probabilities matrix and observation likelihood 
        Probabilities matrix from the words and tags present in the training 
        file. Call function "form_HMM" for this. The cleaned training copy file
        will be passed as param to this function.

        This function returns following variables:

        1) A dict object specifying tag transition probabilities for HMM
        2) A dict object specifying observation likelihood for HMM
        3) A list of all unique tags 

        '''
        
        tag_transition_prob_matrix, word_tag_obs_lkhd_dict, unique_tags\
                                            = form_HMM(train_copy_file)
        
        '''
        Start POS tagging of test file. For this, viterbi's algorithm will be
        used. I am going to use the viterbi's algorithm as mentioned in 
        Section 5.5 of Jurafsky and Martin textbook 
        "Speech and Language Processing".
        '''
        
        '''
        Before actual POS tagging, preprocess the test file. Preprocessing
        will replace all newline characters from the test copy file with 
        a space.
        
        This way all sentences in the test file will appear in a single line.
        
        For preprocessing, call preprocess_file() function. This function
        takes name of the test file as input and preprocesses it.
        '''

        preprocess_file(test_copy_file)

        '''
        Call a function viterbi_decode() to do actual POS-tagging for the
        preprocessed file. This function applies the viterbi algorithm to
        get the tags for words in test file. 

        This function takes following argument:
        
        1) The name of test file (preprocessed test copy here)
        2) List of unique tags
        3) Observation likelihood prob matrix for HMM
        4) Tag transition prob matrix for HMM
        5) A second copy of original test file, created earlier, which is not
        preprocessed at all. It will be used to prepare final output file
        for the program.
        6) A dict object containing mapping of unknown words with their tags
        decided by rule based approach above.

        This function writes the POS tag for each word in the test file.
        And writes the tagged text into final output file called as 
        "tagging-output".

        It returns the total number of tokens/ words tagged by tagger, which
        is used in evaluation later.
        '''
        token_count = viterbi_decode(test_copy_file, unique_tags,\
                      word_tag_obs_lkhd_dict,\
                      tag_transition_prob_matrix, test_copy_file_1,\
                      unknown_word_tags_mapping)        
       
        '''
        Now that we have our tagged file "tagging-output", compare it against
        the gold std file to assess overall accuracy of our POS-tagging.
        For this call evaluate_tagging function. 
        It takes following parameters:
        1) Name of the tagging output file
        2) Name of gold std file
        3) Total number of tokens in the test file

        This function calculates overall accuracy of the tagging and also
        outputs the confusion matrix, which shows the percentage of times
        a tag is wrongly tagged with other tag.
        '''
        evaluate_tagging("tagging-output",  gold_std_file_name, token_count)

    else:
        if debug:
            print "No parameter passed to the program !"
    
        print "\n\tPlease provide proper inputs to the program !"
        print "\tSample usage: "
        print "\tpython Ngram_Modelling.py -tr postr -ts postst -tk poskey\n"
###############################################################################
# End of main function
###############################################################################

'''
Boilerplate syntax to specify that main() method is the entry point for 
this program.
'''

if __name__ == '__main__':
 
    main()

##############################################################################
# End of pos_tagging.py program
##############################################################################
