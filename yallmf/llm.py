'''
Scaffolding to make it easier to work with LLMs
'''

import re
import string
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

    def validate_clean(self, completion):
        '''
        Return only the message
        '''
        pass


@dataclass(kw_only=True)
class ChoicesListPrompt(Prompt):
    '''
    Choose one or more values from a list
    '''
    choices: list = field(default_factory=list)
    doc_tag: str ='\n\n%%DOCUMENT:\n'
    choices_tag: str ='\n\n%%CHOICES:\n'
    result_tag: str = '\n\n%%RESULT:\n'
    strip_punct: bool = True
    lowercase: bool = True
    correction_map: dict = field(default_factory=dict)
    skipvalues: set = field(default_factory=set)
    temperature: float = 0.1

    def validate_clean(self, completion):
        '''Return list of valid values, asking the LLM to correct an invalid value if possible'''

        v = completion['choices'][0]['message']['content']
        vals = [val.strip() for val in v.split(',')]
        if self.strip_punct:
            vals = [val.strip(string.punctuation) for val in vals]
        if self.lowercase:
            vals = [val.lower() for val in vals]
        final = set()
        # correct values not in `choices`
        for v in vals:
            if v in self.choices:
                final.add(v)
            else:
                if v in self.skipvalues:
                    continue
                if v in self.correction_map.keys():
                    final.add(self.correction_map[v])
                else: # ask for a new value n times
                    maxtries=5
                    i=0
                    while i<maxtries:
                        print(f'fixing choice "{v}"...')
                        prompt = f'''You are a machine that chooses a single item from a list. What item from CHOICES most closely matches "{v}"? ONLY RETURN THE MATCHING "CHOICES" VALUE BY ITSELF.''' \
                            +self.choices_tag+', '.join(self.choices)+'\n\n%%CHOICE:'
                        comp = chat_completion(prompt,timeout=10)
                        newv = comp['choices'][0]['message']['content'].strip()
                        newv = newv.strip(string.punctuation)
                        if self.lowercase:
                            newv = newv.lower()
                        if newv in self.choices:
                            print(f'Successfully matched {v} to {newv}')
                            final.add(newv)
                            self.correction_map[v]=newv
                            break
                        else:
                            print(f'BAD: {newv}')
                            i+=1
                    if i==maxtries:
                        print(f'Related lookup not successful after {maxtries} tries. Adding "{v}" to `skipvalues` set.')
                        self.skipvalues.add(v)
        return sorted(list(final))
    def get_prompt(self, doc):
        return self.base_prompt+self.doc_tag+doc+self.choices_tag+', '.join(self.choices)+self.result_tag
    def process_doc(self, doc):
        prompt = self.get_prompt(doc)
        completion = chat_completion(prompt)
        return self.validate_clean(completion)
    def process_docs_gen(self, doclist):
        pass



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
