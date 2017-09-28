import asyncio
from collections import OrderedDict
from collections import defaultdict
import json
import os
import requests as r

from etsy import config
from utils.multi_downloader import AsyncDownloader

api_key = config.get('etsy', 'API_KEY')


def get_all_stores():
    """ Get all etsy stores

    Returns:
         dict: a dictionary of shop name, shop id, and count of active listings
    """
    shops_url = "https://openapi.etsy.com/v2/shops?api_key={}".format(api_key)
    req = r.get(shops_url)
    res = req.json()
    count = int(res["count"])  # total store count
    offsets = range(0, count + 10000, 10000)
    # generate urls to get data from 10000 stores at a time, if a list of stores not provided
    # if list of stores provided, get data for only those stores
    urls = ['https://openapi.etsy.com/v2/shops?api_key=' + api_key + '&limit=10000&offset=' + str(x)
            for x in offsets]

    _results, _stores = [], {}  # hold request results, and parsed store data, respectively
    loop = asyncio.get_event_loop()
    downloader = AsyncDownloader(urls, _results)
    loop.run_until_complete(downloader.results)  # download all urls asynchronously

    for result in _results:
        for x in result["results"]:
            if x["listing_active_count"] > 10:
                _stores[x["shop_name"]] = {"id": x["shop_id"], "active_count": x["listing_active_count"]}

    _sorted_stores = OrderedDict(sorted(_stores.items(), key=lambda x: x[1]["active_count"], reverse=True))

    return _sorted_stores


def get_store_listings(n, sorted_stores):
    """
    Get the top 10 stores sorted by count of listings

    Args:
        n: number of stores (ordered by listing count) to get data
    """

    # top = {x[1]["id"]: [] for x in list(sorted_stores.items())[:10]}

    # generate the urls (with offsets) to get batches of stores listins,
    # for each of the top n stores with the most listings
    urls = {}
    for store in list(sorted_stores.items())[:n]:
        active_count = store[1].get("active_count", 0)
        store_urls = ["https://openapi.etsy.com/v2/shops/" + str(store[0]) + "/listings/active?api_key=" + api_key +
                      "&limit=100&offset=" + str(x) for x in range(0, active_count + 100, 100)]
        urls[str(store[0])] = store_urls
    # get all listings for each of the top n stores
    stores_listings = defaultdict(list)
    for s in urls:
        _results, _stores = [], {}
        downloader = AsyncDownloader(urls[s], _results)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(downloader.results)
        for result in _results:
            for x in result["results"]:
                stores_listings[s].append({"title": x["title"], "description": x["description"]})
    return stores_listings


if __name__ == "__main__":
    stores = get_all_stores()
    store_listings = get_store_listings(20, stores)
    data_path = os.path.join(os.getcwd(), "..", "data", "listings_text.json")
    os.remove(data_path)
    json.dump(store_listings, open(data_path, "w"))
