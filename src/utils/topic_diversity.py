def topic_diversity(model, top_n=10, model_type='lda', feature_names=None):
    """
    Compute topic diversity score for different topic modeling approaches.

    Parameters:
    - model: The trained topic model (LDA, LSA, NMF, Top2Vec, or BERTopic).
    - top_n: Number of top words per topic to consider.
    - model_type: Type of model ('lda', 'nmf', etc.)
    - feature_names: Required for NMF – list of vocabulary terms from TF-IDF vectorizer.

    Returns:
    - diversity_score: Ratio of unique words to total words across topics.
    """
    words = []

    if model_type.lower() == 'lda':
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
        words = [word for topic in topics for word, _ in topic[1]]

    elif model_type.lower() == 'nmf':
        if feature_names is None:
            raise ValueError("For NMF, 'feature_names' (from TfidfVectorizer) must be provided.")

        for topic_idx, topic_weights in enumerate(model.components_):
            top_indices = topic_weights.argsort()[::-1][:top_n]

            top_words = [feature_names[i] for i in top_indices]

            words.extend(top_words)

    elif model_type.lower() == 'lsa':
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
        words = [word for topic in topics for word, _ in topic[1]]

    elif model_type.lower() == 'top2vec':
        topic_words, _, _ = model.get_topics()
        for topic in topic_words:
            top_topic_words = topic[:top_n]
            for phrase in top_topic_words:
                # Split multi-word phrases into individual words
                split_words = phrase.split()
                words.extend(split_words)

    elif model_type.lower() == 'bertopic':
        topic_ids = [topic_id for topic_id in model.get_topics().keys() if topic_id != -1]
        words = []
        for topic_id in topic_ids:
            topic_words = model.get_topic(topic_id)[:top_n]
            words.extend([word for word, _ in topic_words])

    else:
        raise ValueError("Unsupported model type. Choose from ['lda', 'lsa', 'nmf', 'top2vec', 'bertopic'].")

    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0

    return diversity_score
