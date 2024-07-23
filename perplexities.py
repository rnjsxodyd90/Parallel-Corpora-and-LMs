en_sentences_test_set = removing_whitespace(test_set, language = "en")
nl_sentences_test_set = removing_whitespace(test_set, language = "nl")
it_sentences_test_set = removing_whitespace(test_set, language = "it")
  
en_bigram_test_sentences = Corpus( t = 20, n = 2, corpus = en_sentences_test_set, bos_eos = True, vocab = vocabulary )
nl_bigram_test_sentences = Corpus( t = 20, n = 2, corpus = nl_sentences_test_set, bos_eos = True, vocab = vocabulary )
it_bigram_test_sentences = Corpus( t = 20, n = 2, corpus = it_sentences_test_set, bos_eos = True, vocab = vocabulary )

en_tetragram_test_sentences = Corpus( t = 20, n = 4, corpus = en_sentences_test_set, bos_eos = True,  vocab = vocabulary )
nl_tetragram_test_sentences = Corpus( t = 20, n = 4, corpus = nl_sentences_test_set, bos_eos = True,  vocab = vocabulary )
it_tetragram_test_sentences = Corpus( t = 20, n = 4, corpus = it_sentences_test_set, bos_eos = True,  vocab = vocabulary )

# Dictionary to group the models

models = {
    "2": {"category": {"sents": bigram_model_sentences,"words": bigram_model_types}, "data": {
           "ENtrain_sents": bigram_training_sentences,
            "ENtrain_words": bigram_training_types,
            "ENtest": en_bigram_test_sentences,
            "NLtest": nl_bigram_test_sentences,
            "ITtest": it_bigram_test_sentences }
    },
     "4": {"category": {"sents": tetragram_model_sentences,"words": tetragram_model_types}, "data": {
            "ENtrain_sents": tetragram_training_sentences,
            "ENtrain_words": tetragram_training_types,
            "ENtest": en_tetragram_test_sentences,
            "NLtest": nl_tetragram_test_sentences,
            "ITtest": it_tetragram_test_sentences #italian testset
        }
    }
}
results = {"ngram_size": [],"training_data": [],"test_data": [],"perplexity": []}
for ngram in models:
    for key in models[ngram]["category"]:
        for data in models[ngram]["data"]:
            model = models[ngram]["category"][key]
            test = models[ngram]["data"][data]
            #Calculating the perplexity for the current model using the specified test data
            perplexity = model.perplexity( test )
            # Appending the results of the perplexity calculation to the appropriate lists in the 'results' dictionar
            results["ngram_size"].append(ngram)
            results["training_data"].append( key )
            results["test_data"].append( data )
            results["perplexity"].append(round(perplexity,4))


df = pd.DataFrame(results)
df

df.to_csv( 'TaeyongKwon_perplexities.csv', encoding='utf-8', index=False)
