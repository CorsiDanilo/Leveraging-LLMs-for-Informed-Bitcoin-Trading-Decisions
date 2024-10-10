import fasttext

class fasttext_lang_model:
    def __init__(self):
        pretrained_lang_model = "models/pretrained/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        # Extract the language from the prediction
        predictions = predictions[0][0].replace("__label__", "")
        
        return predictions