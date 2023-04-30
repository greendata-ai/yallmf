'''
Interact with OpenAI API
'''

import openai
from .utils import make_request, run_with_timeout


def chat_completion(msg,max_tokens=300,model="gpt-3.5-turbo",timeout=5*60):
    '''
    Get OpenAI chat completion
    '''
    result = run_with_timeout(openai.ChatCompletion.create,
        model=model,
        messages=[
                        #{"role": "system", "content": systemprompt},
                        {"role": "user", "content": msg}
                    ],
        max_tokens=max_tokens,
        timeout=timeout
        )

    return result