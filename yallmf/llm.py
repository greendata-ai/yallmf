'''
Scaffolding to make it easier to work with LLMs
'''

from enum import Enum
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np
from .openai_api import chat_completion

WORDS_PER_TOKEN = 4
@dataclass
class Prompt:
    base_prompt: str
    max_length: int = 4096*WORDS_PER_TOKEN # for chat-gpt-3.5;
    # gpt-4 8192*4, gpt-4-32k 32768*4
    temperature: float = 0.5

@dataclass(kw_only=True)
class JsonPrompt(Prompt):
    '''
    Generate JSON that associates attributes to each 
    document. The prompt should return completions
    of the format
    {ID: {attribute:value}} OR {ID: value}
    '''
    doc_tag: str ='\n\n%%DOCUMENT:'
    json_tag: str = '\n\n%%JSON\n\n'
    max_retries_invalid_completion: int = 3
    num_lentest_docs: int = 5
    return_doc_len: int = 700
    temperature: float = 0.1
    _test_lens: list = field(default_factory=list)
    _retried_invalid: int = 0

    def validate_clean(self, completion):
        '''
        Validate and clean completion, returning a dict
        in standardized format. Return None if incorrect
        '''
        
        pass
        # null return if invalid
    
    def get_prompt(self, docs):
        '''
        Return a single prompt (for a subset of docs)
        '''
        prompt = self.base_prompt+'\n\n'
        for i,d in enumerate(docs):
            prompt += self.doc_tag+f'{i}\n\n'+d
        prompt += self.json_tag
        return prompt
    def doc_completions_gen(self, doclist):
        '''
        A generator yielding {attr:value} dictionaries 
        parsed from the JSON completion.

        It will batch several documents at a time, attempting
        to stay under the `max_length` of a completion.
        '''
        return_doc_len = self.return_doc_len

        current_docs=[]
        lastprompt = None
        for i,d in enumerate(doclist):
            current_docs.append(d)
            newprompt = self.get_prompt(current_docs)
            # if it's too long OR it's the last doc:
            if (len(newprompt)+len(current_docs)*return_doc_len>self.max_length
                ) or i==len(doclist)-1:
                # run the last prompt
                max_tokens=int(self.return_doc_len*len(current_docs)/WORDS_PER_TOKEN)
                _prompt=None
                if lastprompt is not None:
                    _prompt=lastprompt
                else:
                    _prompt=newprompt
                completion = chat_completion(_prompt,
                    max_tokens=max_tokens,model='gpt-4')
                # make sure these are ordered right?
                # how to handle redos? Maybe need to do a batch version of this
                for retdoc in self.validate_clean(completion):
                    raise # TODO: finish this
                    yield retdoc
                lastprompt=None
                current_docs=[]

def similarity(embeddings, target_embedding):
    '''
    Return an array of cosine similarity scores for similarity
    of `target_embedding` to each of `embeddings`
    '''
    # convert embeddings to np.array
    if isinstance(embeddings,pd.Series):
        Y=np.array(embeddings.values.tolist())
    elif isinstance(embeddings,list):
        Y=np.array(embeddings)
    else:
        Y=embeddings

    sim = cosine_similarity(embeddings, [target_embedding]
        ).reshape(-1)
    return sim
