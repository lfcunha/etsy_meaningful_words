import asyncio
import requests as r


class AsyncDownloader(object):
    """Downloads a list of urls asynchronously, collecting the results in a list within the class

    """
    def __init__(self, urls, results):
        self._results = results
        self._urls = urls

    @property
    def results(self):
        return self.download()

    @asyncio.coroutine
    def download(self):
        """ Asynchronously query multiple urls

        """

        loop = asyncio.get_event_loop()

        futures = [loop.run_in_executor(None, r.get, url) for url in self._urls]
        responses = {}

        for x in range(len(self._urls)):
            responses[str(x)] = yield from futures[x]

        for key in responses.keys():
            try:
                self._results.append(responses[key].json())
            except Exception as e:  # requests results with offset past the total results count. response will contain no json
                print(e, key, responses[key.text])

        return self._results
