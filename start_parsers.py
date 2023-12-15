import sys
from project.machine_learning.parsing import DataParser

def main(): 
    args = sys.argv
    print("Start parsing",args[1])
    parser = DataParser(args[1])
    parser.parse(True)

if __name__ == '__main__' :
    main()