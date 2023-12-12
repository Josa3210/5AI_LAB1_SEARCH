from project.machine_learning.parsing import DataParser

def main(): 
    parser = DataParser("project/data/Carlsen.pgn")
    parser.parse(True)

if __name__ == '__main__' :
    main()