import os
import string
import argparse

def change_year(path, keyword, new_year=""):
    for file in os.listdir(path):
        results = []
        if keyword in file:
            print(f'found "{file}"')
            year=file[:5]
            new_name = file.replace(year,new_year)
            print(f'change to {new_name}')
            os.rename(os.path.join(path,file), os.path.join(path,new_name))
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename Files')
    parser.add_argument('path', metavar='PATH', type=str,help='Input file directory')
    parser.add_argument('keyword', metavar='PATH', type=str,help='File names that contains the keyword')
    parser.add_argument('new_year', metavar='new year', type=str, help='New year')
    input_dir = parser.parse_args().path
    new_years = parser.parse_args().new_year
    keyword   = parser.parse_args().keyword
    change_year(input_dir, keyword,new_years)
    