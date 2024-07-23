# Removing whitespaces and saving to list

def removing_whitespace(corpus, language = "en"): 
    list_of_lists = []
    for ID in corpus:
        for sentence in corpus[ID][language]:
            sentence = list( re.sub(r'\s', '', sentence) )
            list_of_lists.append( sentence )
                
    return list_of_lists

sentences_train_set = removing_whitespace(train_set)


# Extracting word types from the sentences and saving each type as lists of its characters

def word_types(corpus, language="en"):
    # Using a set comprehension to collect unique words across all sentences in the specified language of the corpus.
    types = {word for ID in corpus for sentence in corpus[ID][language] for word in sentence.split()}
    
    # Converting each word in the set to a list of characters and return as a list of these lists.
    return [list(word) for word in types]


word_types_train_set = word_types(train_set)

word_types_train_set[:4]

class Corpus(object):

    def __init__(self, t, n, corpus = None, bos_eos = True, vocab = None):
    
        """        
        A Corpus object has the following attributes:
         - corpus: list of lists of characters..
         - vocab: set of unique characters 
         - t: int. Words with a lower frequency count than t are replaced with the UNK string
         - ngram_size: int, 2 for bigrams, 3 for trigrams, and so on.
         - bos_eos: bool, default to True. If False, bos and eos symbols are not prepended and appended to sentences.
         - frequencies: Counter, mapping tokens to their frequency count in the corpus
        """
        
        self.vocab = vocab        
        self.sentences = corpus
        self.t = t
        self.ngram_size = n
        self.bos_eos = bos_eos        
        self.frequencies = self.freq_distr()
                
        if self.t or self.vocab:
            self.sentences = self.filter_words()
                        
        if self.bos_eos:
            self.sentences = self.add_bos_eos()                            
    
    def freq_distr(self):
         return Counter([word for sentence in self.sentences for word in sentence])
        
    
    def filter_words(self):
        
        """
        Replaces illegal tokens with the UNK string. A token is illegal if its frequency count
        is lower than the given threshold and/or if it falls outside the specified vocabulary.
        """
        
        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                if self.t and self.vocab:
                    # check that the word is frequent enough and occurs in the vocabulary
                    filtered_sentence.append(
                        word if self.frequencies[word] > self.t and word in self.vocab else 'UNK'
                    )
                else:
                    if self.t:
                        # check that the word is frequent enough
                        filtered_sentence.append(word if self.frequencies[word] > self.t else 'UNK')
                    else:
                        # check if the word occurs in the vocabulary
                        filtered_sentence.append(word if word in self.vocab else 'UNK')
            
            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)
    
        return filtered_sentences
    
    def add_bos_eos(self):
        
        """
        Adds the necessary number of BOS symbols and one EOS symbol.
        
        In a bigram model, you need on bos and one eos; in a trigram model you need two bos and one eos, and so on...
        """
        
        r = 1 if self.ngram_size == 1 else self.ngram_size - 1
        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = ['#bos#']*r + sentence + ['#eos#']
            padded_sentences.append(padded_sentence)
    
        return padded_sentences
      
  class LM(object):
    
    def __init__(self, n, vocab = None, smoother = 'Laplace', k = 0.01, lambdas = None):
        
        self.vocab = vocab
        self.k = k if smoother == 'Laplace' else 0
        self.ngram_size = n
        self.smoother = smoother
        self.lambdas = lambdas if lambdas else {i+1: 1/n for i in range(n)}
        
    def get_ngram(self, sentence, i, n):
        
        if n == 1:
            return sentence[i]
        else:
            ngram = sentence[i-(n-1):i+1]
            history = tuple(ngram[:-1])
            target = ngram[-1]
            return (history, target)
        
                    
    def update_counts(self, corpus, n):
        
        if self.ngram_size != corpus.ngram_size:
            raise ValueError("The corpus was pre-processed considering an ngram size of {} while the "
                             "language model was created with an ngram size of {}. \n"
                             "Please choose the same ngram size for pre-processing the corpus and fitting "
                             "the model.".format(corpus.ngram_size, self.ngram_size))
        
        self.counts = defaultdict(dict)
        # if the interpolation smoother is selected, then estimate transition counts for all possible ngram_sizes
        # smaller than the given one, otherwise stick with the input ngram_size
        ngram_sizes = [n] if self.smoother != 'Interpolation' else range(1,n+1)
        for ngram_size in ngram_sizes:
            self.counts[ngram_size] = defaultdict(dict) if ngram_size > 1 else Counter()
        for sentence in corpus.sentences:
            for ngram_size in ngram_sizes:
                for idx in range(n-1, len(sentence)):
                    ngram = self.get_ngram(sentence, idx, ngram_size)
                    if ngram_size == 1:
                        self.counts[ngram_size][ngram] += 1
                    else:
                        # it's faster to try to do something and catch an exception than to use an if statement to 
                        # check whether a condition is met beforehand. The if is checked everytime, the exception 
                        # is only catched the first time, after that everything runs smoothly
                        try:
                            self.counts[ngram_size][ngram[0]][ngram[1]] += 1
                        except KeyError:
                            self.counts[ngram_size][ngram[0]][ngram[1]] = 1
        
        
        # first loop through the sentences in the corpus, than loop through each word in a sentence
        if self.vocab == None:
            self.vocab = {word for sentence in corpus.sentences for word in sentence}
        self.vocab_size = len(self.vocab)
        
    
    def get_unigram_probability(self, ngram):
        
        tot = sum(list(self.counts[1].values())) + (self.vocab_size*self.k)
        
        try:
            ngram_count = self.counts[1][ngram] + self.k
        except KeyError:
            ngram_count = self.k
        
        return ngram_count/tot
    
    def get_interpolated_ngram_probability(self, history, target):
                
        probability = self.get_unigram_probability(target)*lambdas[1]
        while len(history) > 0:
            n = len(history) + 1
            try:
                c = self.counts[n][history][target]
                t = sum(list(self.counts[n][history].values()))
                probability += c / t * self.lambdas[n]
            except KeyError:
                probability += 0
            history = history[1:]
        
        return probability
    
    def get_laplace_ngram_probability(self, history, target):
        
        try:
            ngram_tot = np.sum(list(self.counts[self.ngram_size][history].values())) + (self.vocab_size*self.k)
            try:
                transition_count = self.counts[self.ngram_size][history][target] + self.k
            except KeyError:
                transition_count = self.k
        except KeyError:
            transition_count = self.k
            ngram_tot = self.vocab_size*self.k
            
        return transition_count/ngram_tot 
    
    def perplexity(self, test_corpus):
        
        probs = []
        for sentence in test_corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx, self.ngram_size)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    if self.smoother == 'Laplace':
                        probs.append(self.get_laplace_ngram_probability(ngram[0], ngram[1]))
                    elif self.smoother == 'Interpolation':
                        probs.append(self.get_interpolated_ngram_probability(ngram[0], ngram[1]))
                        
        entropy = np.log2(probs)

        # this assertion makes sure that valid probabilities are retrieved, whose log must be <= 0
        assert all(entropy <= 0)
        
        
        avg_entropy = -1 * (sum(entropy) / len(entropy))
        
        return pow(2.0, avg_entropy)
    
    def generate(self, limit):
    
        i = 0
        r = 1 if self.ngram_size == 1 else self.ngram_size - 1
        sentence = ['#bos#']*r
        current = sentence[-(self.ngram_size-1):]
    
        while i < limit:
        
            # create a vector of the possible words with relative probabilities: for the unigram model, just 
            # take each unigram probability, for ngram models of higher orders, condition on the current ngram.
            words = []
            probabilities = []
            continuations = self.counts[self.ngram_size] if n == 1 else self.counts[self.ngram_size][tuple(current)]
            tot = sum(list(continuations.values()))
            for w, v in continuations.items():
                words.append(w)
                probabilities.append(v/tot)
        
            # generate a new token according to the probabiity distribution
            new = np.random.choice(words, size=1, p=probabilities)[0]
        
            # stop generating if we hit an end of sequence token.
            if new != '#eos#': 
                sentence.append(new) 
            else: 
                return ' '.join(sentence[n-1:])
        
            # update the current ngram to proceed generating and increment the counter so that we don't keep 
            # generating forever and we can stop if we hit the maximum value we provided as input
            current = sentence[-(n-1):]   
            i += 1
    
        # return the generated sentence if no end of sequence symbol is generated.
        return ' '.join(sentence[n-1:])


# preparing  corpora

# Bigram and Tetragram Corpora for Sentence-Based Training

bigram_training_sentences = Corpus(t=20, n=2, corpus=sentences_train_set, bos_eos=True, vocab=None)


# Initializing language models

# Bigram Models

bigram_model_sentences = LM(n=2, k=0.01, vocab=None)
bigram_model_sentences.update_counts(bigram_training_sentences, bigram_training_sentences.ngram_size)

vocabulary = bigram_model_sentences.vocab
vocabulary.add("D")

# Tetragram Models

tetragram_model_sentences = LM(n=4, k=0.01, vocab=vocabulary)
tetragram_model_types = LM(n=4, k=0.01, vocab=vocabulary)
bigram_model_types = LM(n=2, k=0.01, vocab=vocabulary)

# updating models with their respective training data

# Bigram Model Updates

bigram_training_types = Corpus(t=20, n=2, corpus=word_types_train_set, bos_eos=True, vocab=vocabulary)

bigram_model_types.update_counts(bigram_training_types, bigram_training_types.ngram_size)
tetragram_training_sentences = Corpus(t=20, n=4, corpus=sentences_train_set, bos_eos=True, vocab=vocabulary)

# Bigram and Tetragram Corpora for Word Type-Based Training

tetragram_training_types = Corpus(t=20, n=4, corpus=word_types_train_set, bos_eos=True, vocab=vocabulary)

# Tetragram Model Updates

tetragram_model_sentences.update_counts(tetragram_training_sentences, tetragram_training_sentences.ngram_size)
tetragram_model_types.update_counts(tetragram_training_types, tetragram_training_types.ngram_size)

# Saving the models

import pickle as pkl

# File paths for the models
bigram_sentences_file = 'TaeyongKwon_sents_2gr_en.pkl'
bigram_types_file = 'TaeyongKwon_words_2gr_en.pkl'
tetragram_sentences_file = 'TaeyongKwon_sents_4gr_en.pkl'
tetragram_types_file = 'TaeyongKwon_words_4gr_en.pkl'

# Saving the bigram and tetragram models
with open(bigram_sentences_file, 'wb') as f_out:
    pkl.dump(bigram_model_sentences, f_out)

with open(bigram_types_file, 'wb') as f_out:
    pkl.dump(bigram_model_types, f_out)

with open(tetragram_sentences_file, 'wb') as f_out:
    pkl.dump(tetragram_model_sentences, f_out)

with open(tetragram_types_file, 'wb') as f_out:
    pkl.dump(tetragram_model_types, f_out)

# File path for the bigram sentences model
bigram_sentences_file = 'TaeyongKwon_sents_2gr_en.pkl'

# Reading the bigram sentences model from the file
with open(bigram_sentences_file, 'rb') as f_in:
    bigram_model_sentences_loaded = pkl.load(f_in)

# checking  data from the loaded model to confirm it's loaded correctly
print(bigram_model_sentences_loaded)  

