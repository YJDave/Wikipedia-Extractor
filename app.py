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
from pyspark import SparkConf, SparkContext

import matplotlib
# For OS X users if your code do not work, uncomment below line:
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import mpld3
import numpy as np

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

@app.route('/extract', methods=['POST'])
def extract_data():
    if request.method == "POST":
        word_frequencies = extract_subsection_content(url, sub_section, text_file_name)
        chart_html_array = draw_chart(word_frequencies)

    return render_template('index.html', chart_html_array=chart_html_array)

def draw_chart(word_frequencies):
    chart_html_array = []

    # BAR CHART: Display bar charts of different words
    objects = tuple(word_frequencies.keys())
    y_pos = np.arange(len(objects))
    performance = list(word_frequencies.values())

    fig, ax = plt.subplots()
    ax.bar(y_pos, performance, align='center', alpha=0.5, color="r")
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('No of occurences')
    plt.title('Bar Chart: Word Frequency')
    chart_html1 = mpld3.fig_to_html(fig)
    chart_html_array.append(chart_html1)

     # SCATTER CAHRT: Display bar charts of different words
    x = [i for i in range(1, len(objects)+1)]
    y = performance
    labels = list(objects)

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y, 'ro')
    # You can specify a rotation for the tick labels in degrees or with keywords.
    plt.xticks(x, labels, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.ylabel('No of occurences')
    plt.title('Scatter Chart: Words Frequency')
    plt.subplots_adjust(bottom=0.15)
    chart_html2 = mpld3.fig_to_html(fig2)
    chart_html_array.append(chart_html2)
    # # plt.show()
    # mpld3.show()

    sorted_by_value = sorted(word_frequencies.items(), key=lambda kv: kv[1])
    fig3, ax3 = plt.subplots()

    max_top_values = -10
    top_values = sorted_by_value[max_top_values:]
    objects3 = [i[0] for i in top_values]
    performance3 = [i[1] for i in top_values]

    # print(top_values, objects3, performance3)
    y_pos3 = np.arange(len(objects3))
    ax3.bar(y_pos3, performance3, align='center', alpha=0.5)
    plt.margins(0.2)
    plt.xticks(y_pos3, objects3)
    plt.ylabel('No of occurences')
    plt.title('Bar Chart: Top Most Frequent Words')
    plt.subplots_adjust(bottom=0.15)
    chart_html3 = mpld3.fig_to_html(fig3)
    chart_html_array.append(chart_html3)

    return chart_html_array

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
    print("Fetching html data from URL..")
    html = urllib.request.urlopen(url)
    # Beautify the HTML content
    print("clean data with BeautifulSoup...")
    soup = BeautifulSoup(html, features="html.parser")

    title = soup.find("h1", {"id": "firstHeading"})

    # Will take the Machine Learning URL
    print("Get wikipedia page...")
    p = wikipedia.page(title.text)

    # Will find the subsection from the given article
    print("Get content of subsection...")
    subsection_content = p.section(sub_section)

    # print(subsection_content)
    print("Store result in file...")
    file = open(text_file_name, "w")
    file.write(subsection_content)
    file.close()

    conf = SparkConf().setMaster("local").setAppName("WikipediaExtraction")
    sc =  SparkContext.getOrCreate(conf=conf)

    text_file_content = sc.textFile(text_file_name)

    ### Read the content of file and count words from it below:

    print("Clearning data and analysing data using PySpark...")
    non_empty_text = text_file_content.filter(lambda x: len(x) > 0)
    words = non_empty_text.flatMap(lambda x: clean_str(x).split(' '))
    word_count = words.map(lambda x:(x, 1)).reduceByKey(lambda x, y: x + y) \
                      .map(lambda x: (x[1], x[0])) \
                      .sortByKey(False)

    result_words = {};
    for word in word_count.collect():
        result_words[word[1]] = word[0]

    return result_words

###

if __name__ == '__main__':
    app.run()
