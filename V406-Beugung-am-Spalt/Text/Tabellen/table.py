import numpy as np
print("Bitte Dateinamen eingeben (ohne Endung): ")
datainput=input()
datafile=datainput+".txt"
datatex=datainput+".tex"
while True:
   array=[]
   i=0
   data=open(datafile, "r")
   for line in data:
       array.extend([None]);
       array[i]=str(line.replace(".", ","))
       i+=1
   data.close()

   data=open(datafile, "w")
   i=0
   while i < len(array):
       data.write(array[i])
       i+=1
   data.close()

   data=np.genfromtxt(datafile, unpack=True, dtype="U12")
   #Länge der Spalten: len(data[i])
   #Länge der Zeilen: len(data)
   table=open(datatex, "w")
   table.write("\\begin{table}[H] \n   \centering \n   \caption{name} \n   \label{tab:name} \n   \\begin{tabular} ")
   i=0
   table.write("{ ")
   while i<len(data):
       table.write("c ")
       i+=1
   table.write("} \n \\toprule \n ")
   i=0
   while i < len(data):
       table.write("{$")
       table.write(data[i][0])
       table.write("\:/\: \mathrm{")
       table.write(data[i][1])
       table.write("}$}")
       if i+1 ==len(data):
           table.write(" \\\ \n")
       else:
           table.write(" & ")
       i+=1
   table.write("    \midrule \n")
   j=2
   while j < len(data[0]):
       i=0
       table.write("    ")
       while i < len(data):
           table.write(data[i][j])
           if i+1 ==len(data):
               table.write(" \\\ \n")
           else:
               table.write(" & ")
           i+=1
       j+=1
   table.write("    \\bottomrule \n  \end{tabular}\n\end{table}")
   table.close()

   i=0
   data=open(datafile, "r")
   for line in data:
       array[i]=str(line.replace(",", "."))
       i+=1
   data.close()

   data=open(datafile, "w")
   i=0
   while i < len(array):
       data.write(array[i])
       i+=1
   data.close()
   print('Done. \n \n')
   print('Bitte weitere Dateinamen eingeben. Um das Programm zu beenden Leerzeichen " " eingeben.')
   datainput=input()
   if datainput==" ":
       print('Programm wird beendet.')
       break
   else:
    datafile=datainput+".txt"
    datatex=datainput+".tex"
