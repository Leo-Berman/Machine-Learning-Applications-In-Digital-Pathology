import csv
import sys
from scipy.fftpack import dct

def main():
    path = 'DATA/dctoutput.csv'

    # Open CSV file in read mode with newline to skip "/n"
    #

    # create a list for each value of RGBA
    #
    red = []
    green = []
    blue = []
    alpha = []

    # Open CSV file in read mode with newline to skip "/n"
    #

    with open(path, 'r', newline='') as file:
        # Create CSV reader object
        data = csv.reader(file)

        for thing in data:
            for row in thing:
                for item in row:
                    if item == "['[']":
                        pass
                    else:
                        red.append(item)
                        green.append(row[1])
                        blue.append(row[2])
                        alpha.append(row[3])


  
    
    print(f"Red: {red}")
    print(f"Green: {green}")
    print(f"Blue: {blue}")
    print(f"Alpha: {alpha}")




if __name__ == "__main__":
    main()





    
