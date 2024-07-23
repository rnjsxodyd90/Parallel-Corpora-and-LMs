# Definition of data structures to hold model configurations and results
Table = {"2": {
        "category": {
            "sents": bigram_model_sentences,
            "words": bigram_model_types
        },
        "language": {
            "nl": nl_frequent_words,
            "it": it_frequent_words
        }
    },
    "4": {
        "category": {
            "sents": tetragram_model_sentences,
            "words": tetragram_model_types
        },
        "language": {
            "nl": nl_frequent_words,
            "it": it_frequent_words
        }
    }
}

highest_perplexity = {
    "lang": [],
    "word": [],
    "ngram_size": [],
    "training_data": [],
    "perplexity": []
}

lowest_perplexity = {
    "lang": [],
    "word": [],
    "ngram_size": [],
    "training_data": [],
    "perplexity": []
}

# Main loop to process each language and model type, compute perplexities, and store results
for ngram, content in Table.items():
    for category_key, model in content["category"].items():
        for lang, words in content["language"].items():
            # Computing perplexity for the current configuration
            min_word, min_perplexity, max_word, max_perplexity = perplexity(words, model)

            # Recording the highest perplexity
            highest_perplexity["lang"].append(lang)
            highest_perplexity["word"].append(max_word)
            highest_perplexity["ngram_size"].append(ngram)
            highest_perplexity["training_data"].append(category_key)
            highest_perplexity["perplexity"].append(max_perplexity)
            
            # Recording the lowest perplexity
            lowest_perplexity["lang"].append(lang)
            lowest_perplexity["word"].append(min_word)
            lowest_perplexity["ngram_size"].append(ngram)
            lowest_perplexity["training_data"].append(category_key)
            lowest_perplexity["perplexity"].append(min_perplexity)


max_df = pd.DataFrame(highest_perplexity)
max_df


min_df = pd.DataFrame(lowest_perplexity)


min_df
