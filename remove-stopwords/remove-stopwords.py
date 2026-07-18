def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    
    result = [word for word in tokens if word not in stopwords]
    return result
    pass