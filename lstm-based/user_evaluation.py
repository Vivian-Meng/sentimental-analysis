import spacy
import torch
from data_processing import TEXT
from model import model
nlp = spacy.load('en_core_web_sm')

def predict_sentimental(model, sentence):
    
    # set the model to evaluation mode
    model.eval()

    # tokenize the sentence, split it from a raw string to a list of tokens
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    # index the tokens by converting them into integer representations from our voc
    #~Vocab.stoi – A collections.defaultdict instance mapping token strings to numerical identifiers.
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    # gets the length of our sequences
    length = [len(indexed)]

    # converts the indexes, python list to pytorch
    tensor = torch.LongTensor(indexed).to(device)

    # add a batch dimension by using unsqueezeing
    tensor = tensor.unsqueeze(1)

    # convert the length into a tensor
    length_tensor = torch.LongTensor(length)

    # squashes the output predictions from a real number (0-1) with sigmoid
    prediction = torch.sigmoid(model(tensor, length_tensor))

    #convert the tensor holding a single value into an integer with item() function
    return prediction.item()



predict_sentimental(model, "This film is terrible")