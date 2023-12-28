def split(filePath: str, destinationPath: str, amountGames: int = 50000):
    whiteLines = 0
    totalGames = 0
    iteration = 0
    with open(filePath, 'r') as inputFile:
        outputFile = open(destinationPath + f"_{iteration}.pgn", 'w')
        line = "start"

        # As long there is no empty line, keep reading
        while line != "":
            line = inputFile.readline()
            outputFile.write(line)

            if line == '\n':
                whiteLines += 1

            if whiteLines == 2:
                totalGames += 1
                whiteLines = 0

                if (totalGames + 1) % amountGames == 0:
                    iteration += 1
                    outputFile.close()
                    outputFile = open(destinationPath + f"_{iteration}.pgn", 'w')
                    print(f"Create file {iteration}: Total games {totalGames + 1}", end='\r')

    print(f"Created file {iteration}: Total games {totalGames + 1}", end='\r')


if __name__ == "__main__":
    split(
        "D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\lichess_db\\lichess_db_2023-11.pgn",
        "D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\training\\lichess_db_2023-11"
    )
    # split("D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\training2\\Carlsen.pgn", "D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\training2\\Carlsen", 1000)
