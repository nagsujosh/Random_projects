from bs4 import BeautifulSoup
import requests
import re

# search term variable will collect the input of your needed graphics gard
search_term = input("What product you want to search for? ")

# Decided to use New Egg website for scraping
url = f"https://www.newegg.com/p/pl?d={search_term}&N=4131"

page = requests.get(url).text

doc = BeautifulSoup(page, "html.parser")

page_number = int(doc.find(class_="list-tool-pagination-text").strong.contents[-1])
items_found = {}
for count in range(1, page_number + 1):
    url = f"https://www.newegg.com/p/pl?d={search_term}&N=4131&page={count}"
    page = requests.get(url).text
    doc = BeautifulSoup(page, "html.parser")

    div = doc.find(class_="item-cells-wrap border-cells items-grid-view four-cells expulsion-one-cell")

    items = div.find_all(text=re.compile(search_term))
    for item in items:
        parent = item.parent
        if parent.name != 'a':
            continue
        link = parent['href']

        next_parent = item.find_parent(class_="item-container")
        price = ""
        if next_parent.find(class_='price-current').strong is not None:
            price = next_parent.find(class_='price-current').strong.string

        if price != "":
            items_found[item] = {"price": int(price.replace(",", "")), "link": link}


sorted_items = sorted(items_found.items(), key=lambda x: x[1]['price'])  # found items have been sorted on ascending order
for item in sorted_items:
    print(item[0])
    print(f"${item[1]['price']}")
    print(item[1]['link'])
    print("-"*100)  # Separating the found items
