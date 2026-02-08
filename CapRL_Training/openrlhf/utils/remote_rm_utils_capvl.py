import itertools
import time
import ray
import requests
import torch
import torch.distributed as dist
import traceback

from openrlhf.utils.logging_utils import init_logger
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5, num_threads=1):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for rotate_idx in range(try_max_times):
        try:

            prompts = data.get('prompts')
            queries = data.get("query")
            labels = data.get("labels")
            assert len(prompts) == len(queries), "Mismatched input lengths"

            def chunk_data(data_list, num_chunks):
                """Evenly split a list into num_chunks sublists"""
                k, m = divmod(len(data_list), num_chunks)
                return [data_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]

            prompt_chunks = chunk_data(prompts, num_threads)
            query_chunks = chunk_data(queries, num_threads)
            labels_chunks = chunk_data(labels, num_threads)

            def send_request(url, prompts, queries, labels, headers, score_key):
                try:
                    start_time = time.time()
                    payload = {'prompts': prompts, 'query': queries, 'labels': labels}
                    response = requests.post(url=url, json=payload, headers=headers, timeout=1800)
                    response.raise_for_status()
                    elapsed_time = time.time() - start_time
                    logger.info(f"Request to {url} took {elapsed_time:.3f} seconds.")
                    result = response.json()
                    assert score_key in result, f"{score_key} not in {result}"
                    return result.get(score_key)
                except Exception as e:
                    logger.error(f"Error with URL {url}: {e}")
                    return None
            #local_rank = (rotate_idx + dist.get_rank()) % 8
            #new_port = str(8888 + local_rank)
            #print (dist.get_rank(), new_port)
            results = [None] * num_threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {
                    executor.submit(
                        send_request,
                        #url=url.replace('8888', str(8888 + i)),
                        url=url,
                        prompts=prompt_chunks[i],
                        queries=query_chunks[i],
                        labels=labels_chunks[i],
                        headers=headers,
                        score_key=score_key
                    ): i for i in range(num_threads)
                }

                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    results[i] = future.result()
                results = list(itertools.chain.from_iterable(results))
                return results

        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
            traceback.print_exc()
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, prompts, labels, score_key="rewards", num_threads=1):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels}, score_key, num_threads=num_threads)
    #return torch.tensor(scores)
    return scores

import os
from contextlib import contextmanager

@contextmanager
def no_proxy():
    """A context manager to temporarily disable proxy settings from environment variables."""
    # List of common proxy environment variables (both lowercase and uppercase)
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    
    # 1. Backup: Store original proxy settings if they exist
    original_proxies = {}
    for key in proxy_keys:
        if key in os.environ:
            original_proxies[key] = os.environ[key]
            
    # 2. Teardown: Remove the proxy variables from the current environment
    if original_proxies:
        print("Proxy environment variables temporarily disabled...")
        for key in original_proxies:
            del os.environ[key]
            
    try:
        # 3. Yield control to the code inside the 'with' block
        #    This is where your code will run without a proxy.
        yield
    finally:
        # 4. Restore: Put the original proxy settings back
        if original_proxies:
            os.environ.update(original_proxies)
            print("Proxy environment variables restored.")


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, score_key="rewards"):
    with no_proxy():
        return remote_rm_fn(api_url, queries, prompts, labels, score_key)
