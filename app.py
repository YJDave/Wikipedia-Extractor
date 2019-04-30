### Imports

# Flask imports
from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request

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
# Note: Don't forget to update .gitignore, if you change text_file_name variable.

url = 'https://en.wikipedia.org/wiki/Machine_learning'
sub_section = "Theory"
text_file_name = "text-data.txt"
app = Flask(__name__)

### Functions

@app.route('/')
def index():
    return render_template('index.html')

# To convert unicode to clean string and remove punctuations
def clean_str(x):
    # To change from unicode to string
    converted = x.encode('utf-8')

    # To lowercase letters
    lowercased_str = x.lower()

    # Remove all the punctuations
    clean_str = lowercased_str.translate(str.maketrans('', '', string.punctuation))
    # print("Clean str ", clean_str)
    return clean_str


def extract_subsection_content(url, sub_section, text_file_name):
    # Extract HTML content from given url
    html = urllib.request.urlopen(url)
    # Beautify the HTML content
    soup = BeautifulSoup(html, features="html.parser")

    title = soup.find("h1", {"id": "firstHeading"})

    # Will take the Machine Learning URL 
    p = wikipedia.page(title.text)

    # Will find the subsection from the given article
    subsection_content = p.section(sub_section)

    # print(subsection_content)
    file = open(text_file_name, "w")
    file.write(subsection_content)
    file.close()

    sc = SparkContext("local", "WikipediaExtractor App")

    text_file_content = sc.textFile(text_file_name)

    ### Read the content of file and count words from it below:

    non_empty_text = text_file_content.filter(lambda x: len(x) > 0)
    words = non_empty_text.flatMap(lambda x: clean_str(x).split(' '))
    word_count = words.map(lambda x:(x, 1)).reduceByKey(lambda x, y: x + y) \
                      .map(lambda x: (x[1], x[0])) \
                      .sortByKey(False)

    for word in word_count.collect():
        print(word)

    return word_count

###

if __name__ == '__main__':
    app.run()
