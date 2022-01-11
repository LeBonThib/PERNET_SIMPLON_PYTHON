from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import re

with urlopen("https://en.wikipedia.org/wiki/Tom_Cruise") as tom:
    bsobj = tom.read()
    target = soup(bsobj, 'lxml')

for link in bsobj.findall('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])

for link in bsobj.fin('div',{'id':'bodyContent'}).findall('a', href = re.compile('^(/wiki/)((?!:).)*$')):
    if 'href' in link.attrs:
        print(link.attrs['href'])