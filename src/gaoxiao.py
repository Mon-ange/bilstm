import csv
import codecs
csvFile = codecs.open('output.csv',"r",'utf-8')
reader = csv.reader(csvFile)

csvWFile = codecs.open("output1.csv","w",'utf-8')
writer = csv.writer(csvWFile)

for item in reader:
    if reader.line_num == 1:
        continue
    if item[1]==':':
        continue
    if item[1].startswith('：')
        item[1]=item[1][1:]

    writer.writerow([item[0],item[1],item[2]])


csvWFile.close()
csvFile.close()