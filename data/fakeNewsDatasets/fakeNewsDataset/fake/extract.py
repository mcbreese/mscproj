import pandas
import os

def append_to_csv(input_string):
    with open("fake.csv", "a") as csv_file:
        csv_file.write(input_string + "\n")

def open_txt(input):
    with open(input) as inp:
        data = list(inp) # or set(inp) if you really need a set
        for data in data:
                data.replace("\n","")
                data.replace("\r","")
                data.replace("\t","")
        data="".join(str(data) for data in data)
        append_to_csv(data)
    

path= "C:\\Users\\thoma\\OneDrive - UWE Bristol\\Masters Project\\DATASETS\\fakeNewsDatasets\\fakeNewsDatasets\\fakeNewsDataset\\fake"
ext='.txt'
for files in os.listdir(path):
    if files.endswith(ext):
        open_txt(files)
    else:
        continue
