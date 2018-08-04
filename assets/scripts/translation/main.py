import sys

USAGE = ("""Name: Google Translator
Usage:
$ translate text_to_translate to_lang from_lang
from_lang is optional
Example:
$ mou_translator "您好" en 
Hello
""")

def main():
    if(len(sys.argv)<3):
        print(USAGE)
        return (1)
    text = sys.argv[1]
    dest = sys.argv[2]
    if(len(sys.argv) > 3):
        src = sys.argv[3]
    else:
        src = "auto"
    

if __name__ == '__main__':
    main();