#!/usr/bin/python

import xmltodict

file1 = open("result.xml","r")
message = file1.read()

Dict_Var = xmltodict.parse(message)['shotdetect']['content']['body']['shots']['shot']

#print Dict_Var[1].items()[1][1]

datafile = open("shot_info.txt",'w')

i=0
for x in Dict_Var:
	print "bulu"
	# print Dict_Var[i].items()[1][1]
	# print Dict_Var[i].items()[3][1]
	fbegin = x.items()[3][1]
	flength = x.items()[1][1]
	wdata = str(fbegin) + "," + str(flength) + "."
	print wdata
	datafile.write(wdata)
	# i+=1
datafile.close()
#print xmltodict.parse(message)['shotdetect']['content']['body']['shots']['shot'][1].items()[1][1]
#print xmltodict.parse(message)['shotdetect']['content']['body']['shots']['shot'][1].items()[3][1]
