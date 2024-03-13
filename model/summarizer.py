from transformers import PegasusForConditionalGeneration, PegasusTokenizer


class Summarizer:
    # instantiating the tokenizer and model
    def __init__(self, model_name='google/pegasus-xsum'):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # function for summarizing text which will create tokens, then feed into the model to generate a summary
    def summarize(self, text):
        tokens = self.tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
        summary_ids = self.model.generate(**tokens)
        summary_output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary_output