import math
from .database import Document, GRAM_LENGTH, Gram, Session


NEIGHBOR_STRENGTH = 0.3
LEARNING_WEIGHT = 0.1


def _split_string(text):
    for i in range(len(text) - GRAM_LENGTH + 1):
        end = i + GRAM_LENGTH
        yield text[i:end]


def _sigmoid(x):
    return math.tanh(x)


def _dsigmoid(y):
    if y <= -1.0:
        y = -.9999
    elif y >= 1.0:
        y = .9999
    return math.atanh(y)


def apply_emotion(gram, anger, interest, joy, trust):
    delta_anger = anger + NEIGHBOR_STRENGTH * (interest - trust) - gram.anger
    delta_interest = (interest + NEIGHBOR_STRENGTH * (anger + joy) -
                      gram.interest)
    delta_joy = interest + NEIGHBOR_STRENGTH * (interest + trust) - gram.joy
    delta_trust = trust + NEIGHBOR_STRENGTH * (joy - anger) - gram.trust

    gram.anger = _sigmoid(_dsigmoid(gram.anger) +
                          LEARNING_WEIGHT * delta_anger)
    gram.interest = _sigmoid(_dsigmoid(gram.interest) +
                             LEARNING_WEIGHT * delta_interest)
    gram.joy = _sigmoid(_dsigmoid(gram.joy) +
                        LEARNING_WEIGHT * delta_joy)
    gram.trust = _sigmoid(_dsigmoid(gram.trust) +
                          LEARNING_WEIGHT * delta_trust)


def extract_text(text):
    session = Session()
    grams = session.query(Gram).filter(Gram.gram.in_(_split_string(text)))

    for gram in grams:
        gram.update_idf()

    sums = dict(
        (k, _sigmoid(sum(getattr(g, k) * g.idf for g in grams))) for k in
        ('anger', 'interest', 'joy', 'trust'))
    session.close()
    return sums


def learn_text(text, values=None):
    session = Session()

    document = session.query(Document).get(text)
    if not document:
        document = Document(text)
    sums = extract_text(text) if not values else values

    for piece in _split_string(text):
        gram = session.query(Gram).get(piece)
        if not gram:
            gram = Gram(piece)
            gram.update_idf()

        apply_emotion(gram, anger=sums['anger'], interest=sums['interest'],
                      joy=sums['joy'], trust=sums['trust'])
        gram = session.merge(gram)
        document.grams.append(gram)

    document = session.merge(document)
    session.commit()
