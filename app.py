### Imports

# To make HTTP request to the Wikipedia article
import urllib.request
# To beautify the content and extract data from it
from bs4 import BeautifulSoup
# Wrapper around Wikipedia API
import wikipedia

### Variables

url = 'https://en.wikipedia.org/wiki/Machine_learning'
sub_section = "Theory"

# Extract HTML content from given url
html = urllib.request.urlopen(url)
# Beautify the HTML content
soup = BeautifulSoup(html, features="html.parser")

title = soup.find("h1", {"id": "firstHeading"})

# Will take the Machine Learning URL 
p = wikipedia.page(title.text)

# Will find the "Theory" subsection from the given Wikipedia article
subsection_content = p.section(sub_section)

print(subsection_content)
