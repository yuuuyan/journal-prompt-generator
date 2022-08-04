import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
from collections import defaultdict

def unzip(pairs):
    """
    "unzips" of groups of items into separate tuples.
    
    Example: pairs = [("a", 1), ("b", 2), ...] --> (("a", "b", ...), (1, 2, ...))
    
    Parameters
    ----------
    pairs : Iterable[Tuple[Any, ...]]
        An iterable of the form ((a0, b0, c0, ...), (a1, b1, c1, ...))
    
    Returns
    -------
    Tuple[Tuples[Any, ...], ...]
       A tuple containing the "unzipped" contents of `pairs`; i.e. 
       ((a0, a1, ...), (b0, b1, ...), (c0, c1), ...)
    """
    return tuple(zip(*pairs))

def process_text(path_to_prompts="prompts.txt"):
    """
    Returns processed text from text corpus at given path.
    """

    with open(path_to_prompts, "rb") as f:
        prompts = f.read().decode()  
        prompts = prompts.lower()

    return prompts

def word_count(prompts):
    """
    Gets Counter item of all word counts in text corpus
    """
    tokens = prompts.split()
    word_counts = Counter(tokens) 

    return word_counts

def normalize(counter):
    """ Convert a `letter -> count` counter to a list 
    of (letter, frequency) pairs, sorted in descending order of 
    frequency.

    Parameters
    -----------
    counter : collections.Counter
        letter -> count

    Returns
    -------
    List[Tuple[str, float]]
       A list of tuples: (letter, frequency) pairs in order
       of descending-frequency
    """

    total = sum(counter.values())
    return [(char, cnt/total) for char, cnt in counter.most_common()]

def train_lm(text, n):
    """ Train character-based n-gram language model.

    Parameters
    -----------
    text: str 
        A string (doesn't need to be lowercased, but corpus will be inputted as such after going through process_text()).
        
    n: int
        The length of n-gram to analyze.

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
        
        {n-1 history -> [(letter, normalized count), ...]}
        
        A dictionary that maps histories (strings of length (n-1)) to lists of (char, prob) 
        pairs, where prob is the probability (i.e frequency) of char appearing after 
        that specific history.

    Examples
    --------
    >>> train_lm("cacao", 3)
    {'ac': [('a', 1.0)],
     'ca': [('c', 0.5), ('o', 0.5)],
     '~c': [('a', 1.0)],
     '~~': [('c', 1.0)]}
    """

    raw_lm = defaultdict(Counter) 
    history = "~" * (n - 1) 

    for char in text:
        raw_lm[history][char] += 1
        history = history[1:] + char

    lm = {history : normalize(counter) for history, counter in raw_lm.items()}
    
    return lm

def generate_letter(lm, history):
    """ Randomly picks letter according to probability distribution associated with 
    the specified history, as stored in your language model.

    Note: returns dummy character "~" if history not found in model.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]] 
        The n-gram language model. 
        I.e. the dictionary: history -> [(char, freq), ...]

    history: str
        A string of length (n-1) to use as context/history for generating 
        the next character.

    Returns
    -------
    str
        The predicted character. '~' if history is not in language model.
    """
    if not history in lm:
        return "~"
    letters, probs = unzip(lm[history])
    i = np.random.choice(letters, p=probs)
    return i
    
def generate_text(lm, nletters=100):
    """ Randomly generates `nletters` of text by drawing from 
    the probability distributions stored in a n-gram language model 
    `lm`.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]]
        The n-gram language model. 
        I.e. the dictionary: history -> [(char, freq), ...]
    
    n: int
        Order of n-gram model.
    
    nletters: int
        Number of letters to randomly generate.

    Returns
    -------
    str
        Model-generated text. Should contain `nletters` number of
        generated characters. The pre-pended ~'s are not to be included. 
    """
    # "~" * (n - 1)
    text = []
    for i in range(nletters):
        c = generate_letter(lm, history)
        text.append(c)
        history = history[1:] + c
    return "".join(text)  
    # </COGINST>