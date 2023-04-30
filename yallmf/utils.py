
import signal
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from openai.error import APIConnectionError, RateLimitError
        
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout reached")

def make_request(url):
    """
    Makes an HTTP GET request to the specified URL, with retry logic
    in case of timeouts or certain HTTP status codes.
    """
    # Define the retry strategy
    retry_strategy = Retry(
        total=5,  # Maximum number of retries
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
        retry_on_exception=lambda ex: isinstance(ex, (
            APIConnectionError, RateLimitError))
    )

    # Create a session object with the retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Make the request
    response = session.get(url)

    # Return the response object
    return response

def run_with_timeout(func, *args, timeout=60, **kwargs):
    '''
    Run `func` and raise after `timeout` seconds without completion
    '''
    # Set the signal handler and a 30-second alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Call the function with its arguments and return its result
        result = func(*args, **kwargs)
    except TimeoutException:
        # Handle the timeout exception
        print("Function timed out after {} seconds".format(timeout))
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)

    return result