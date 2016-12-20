Fatima's Spam Filter, 2016-11-30


THE PURPOSE
___________________

The purpose of this program, written in Python, is being a spam filter.
It solves the binary task of sorting what email is spam or "ham" (i.e. good/wanted email).

The program has an amount of emails in text files as an input that are loaded and processed.
Then the features of ham and spam emails are defined in order to finally be classified
with three different classifiers. The user gets an report on the accuracy of each and one of them
and also the top 20 words that are typical for spam emails. 

___________________

GENERAL USAGE NOTES

This spam filter can be applied on both other folders of the same data pool (Enron emails) 
or your own emails. The file directory can be easily changed in the main-part of the code. 
Other than that there are nothing that is required of the user, but compiling the code.

This spam filter is dependent on several modules from the NLTK-package:
the stopwords corpus, WordNetLemmatizer, word_tokenize and all three classifiers.
All of that needs to be in place in order for the program to run smoothly. 


___________________

AUTHOR

Fatima Guseinova
E-mail: fatima.guseinova@ling.su.se


___________________

LICENSE

NLTK license: Apache 2.0 
https://www.apache.org/licenses/LICENSE-2.0
