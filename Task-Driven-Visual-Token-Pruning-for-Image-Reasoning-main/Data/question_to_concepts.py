def extract_concepts(question):

    question = question.lower()

    stop_words = {
        "what","where","when",
        "who","why","how",
        "is","are","the","a","an",
        "of","in","on","to",
        "does","do","did"
    }

    words = question.replace("?","").split()

    concepts = []

    for w in words:
        if w not in stop_words:
            concepts.append(w)

    return concepts


print(
    extract_concepts(
        "What color is the dog?"
    )
)