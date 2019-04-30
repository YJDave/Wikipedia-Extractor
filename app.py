### Imports

# To make HTTP request to the Wikipedia article
import urllib.request
# To beautify the content and extract data from it
from bs4 import BeautifulSoup
# Wrapper around Wikipedia API
import wikipedia

# For word count using PySpark
import re, string
from pyspark import SparkContext

### Variables

url = 'https://en.wikipedia.org/wiki/Machine_learning'
sub_section = "Theory"
text_file_name = "text-data.txt"
word_file_name = "word-count-data.txt"

# Extract HTML content from given url
html = urllib.request.urlopen(url)
# Beautify the HTML content
soup = BeautifulSoup(html, features="html.parser")

title = soup.find("h1", {"id": "firstHeading"})

# Will take the Machine Learning URL 
p = wikipedia.page(title.text)

# Will find the "Theory" subsection from the given Wikipedia article
subsection_content = p.section(sub_section)

# print(subsection_content)
file = open(text_file_name, "w")
file.write(subsection_content)
file.close()

sc = SparkContext("local", "WikipediaExtractor App")

text_file_content = sc.textFile(text_file_name)

### Read the wikipedia content file and count words from it below:

non_empty_text = text_file_content.filter(lambda x: len(x) > 0)
words = non_empty_text.flatMap(lambda x: x.split(' '))
word_count = words.map(lambda x:(x, 1)).reduceByKey(lambda x, y: x + y) \
                  .map(lambda x: (x[1], x[0])) \
                  .sortByKey(False)

for word in word_count.collect():
    print(word)
