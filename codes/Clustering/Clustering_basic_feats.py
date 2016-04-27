# This code takes as input the average image and its LBP image as input
# extracted using the cpp code, and runs clustering on histogram of the pixel intensities. 
# Results are stored in "results/"


import numpy as np
import os
from sklearn import cluster
from PIL import Image, ImageMath

def K_int(X,Y):
#intersection kernel
	return np.sum(np.minimum(X,Y))

def K_lap(X,Y):
#laplacian kernel, tweak with value of gamma
	gamma = 1.0/10000
	return 2.718282**(-gamma*np.sum(np.abs(X-Y)))

#This function is used to match labels with clusters to find the accuracy of the clustering method
def get_max(cluster_Matrix):
	maxVal = 0
	fl = cluster_Matrix.flatten()
	if cluster_Matrix.shape[0] == 1 or cluster_Matrix.shape[1] == 1:
		maxVal = max(fl)
	elif all(fl[x]==fl[0] for x in range(fl.shape[0]) ) :
		maxVal = fl[0] * min([cluster_Matrix.shape[0], cluster_Matrix.shape[1]])
	else:
		val = np.amax(fl)
		ls = []
		for i in range(cluster_Matrix.shape[0]):
			for j in range(cluster_Matrix.shape[1]):
				if cluster_Matrix[i,j] == val:
					ls.append([i,j])
		lsmx = []
		for item in ls:
			l = np.delete(cluster_Matrix,item[0],0)
			l = np.delete(l,item[1],1)
			mx = val + get_max(l)
			lsmx.append(mx)
		maxVal = max(lsmx)
	return maxVal

if __name__ == '__main__':

	no_of_Clusters = 4
	labelIspresent = True
	no_of_characters = 4


	pixels_not_hist = False
	# Set the above as True to use pixels of the flattened image as input features
	# Set the above as False to use histogram of pixels of image as input features

	ls_Track_avg_img_list = os.listdir("avgimgdata/")
	ls_Track_avg_img_list.sort()
	print ls_Track_avg_img_list
	print len(ls_Track_avg_img_list)
	ls_LBP_of_avg_img_list = []

	for filename in ls_Track_avg_img_list:
		fn3 = "1" + filename[1:]		#for LBP image 
		ls_LBP_of_avg_img_list.append(fn3)

	ls_Features = []
	for i in range(len(ls_Track_avg_img_list)):

		filename = "LBPofavgImg/" + ls_LBP_of_avg_img_list[i]
#	Comment the line above and uncomment the line below, if you want to run Clustering on histogram of pixel intensities 
#	of Raw Avg image instead of the LBP image
	#	filename = "avgimgdata/" + ls_LBP_of_avg_img_list[i]

		np_image_filename = np.array(Image.open(filename))
		if pixels_not_hist == True:
			ls_Features.append(np_image_filename.flatten())
		else:
			#Split the image in 3 channels
			#for each channel split image into 4x4 section
			#for each section calculate Histogram of pixel intensities
			#concatenate all the histograms and use this as feature
			ls_npImage_channels = np.split(np_image_filename,3,2)
			ls_Histograms = []
			for X in ls_npImage_channels:
				lsll = np.hsplit(X,4)
				ls_npImage_sections = []
				for XX in lsll:
					ls2 = np.vsplit(XX,4)
					ls_npImage_sections.extend(ls2)
		#			print len(ls_npImage_sections)
				for ZZ in ls_npImage_sections:
					hs, hh = np.histogram(ZZ, bins=128, range=(0,256))
					ls_Histograms.append(hs)
			np_Concatenated_Histograms = np.array(ls_Histograms)
	#		print "dimn of np_Concatenated_Histograms: " + str(np_Concatenated_Histograms.shape)
			ls_Features.append(np_Concatenated_Histograms.flatten())


	np_Features = np.array(ls_Features)
	print "size of np_Features: " + str(np_Features.shape)

	#model = cluster.KMeans(no_of_Clusterslusters=no_of_Clusters, max_iter=300,n_init=100,init="k-means++",)
	model = cluster.SpectralClustering(no_of_Clusterslusters=no_of_Clusters, eigen_solver=None, random_state=None, n_init=100, gamma=1.0, affinity='linear', n_neighbors=10, eigen_tol=0.0, assign_labels='discretize', degree=10, coef0=1, kernel_params=None)

	print "Start fitting into clusters"
	ls_Cluster_labels_predicted = model.fit_predict(np_Features)
	print "Done."
	np_Cluster_labels_predicted = np.array(ls_Cluster_labels_predicted)
	print np_Cluster_labels_predicted.shape
	print np_Cluster_labels_predicted

	for i in range(0,no_of_Clusters):
		newpath = "resultsAVG/" + str(i) + "/"
		if not os.path.isdir(newpath):
			os.makedirs(newpath)

	while i<np_Cluster_labels_predicted.shape[0]:
		grupNo = np_Cluster_labels_predicted[i]
		newpath = "resultsAVG/" + str(grupNo) + "/"
	   	filename = "avgimgdata/" + ls_Track_avg_img_list[i]
	   	image_array = np.array(Image.open(filename))
		ss = newpath + str(i) + ".jpg"
		im = Image.fromarray(image_array)
		im.save(ss)
		i+=1

	##############if label is present#####################################
	if labelIspresent == False:
		exit()

	labels = open("labels.txt","r")
	s = labels.read()
	lb_y = s.split(',')
	del lb_y[-1]

	i=0
	files_list = []

	for i in range(0,no_of_Clusters):
		newpath = "resultsAVG/" + str(i) + "/"
		ss = newpath + str(i) + ".txt"
		fp = open(ss,"w")
		files_list.append(fp)


	i=0
	while i<np_Cluster_labels_predicted.shape[0]:
		grupNo = np_Cluster_labels_predicted[i]
		ss2 = str(lb_y[i]) + ","
		files_list[grupNo].write(ss2)	
		i+=1	
	print "Done. Making the Cluster-Matrix: "

	for i in range(0,no_of_Clusters):
		files_list[i].close()


	diff_labels = []
	for c in lb_y:
		found = False
		for d in diff_labels:
			if c==d:
				found = True
				break
		if found == False:
			diff_labels.append(c)
	print "No. of chars in original data: " + str(len(diff_labels))
	diff_labels.sort()
	cluster_Matrix = np.zeros([no_of_Clusters,len(diff_labels)], dtype=int)

	i=0
	for i in range(0,no_of_Clusters):
		ssn = "resultsAVG/" + str(i) + "/" + str(i) + ".txt"
		print ssn
		clust_file = open(ssn,"r")
		data = clust_file.read()
		clust_file.close()
		print data
		stringdata = data.split(",")
		del stringdata[-1]
		print stringdata
		for k in stringdata:
			j=0
			for l in diff_labels:
				if k==l:
					cluster_Matrix[i][j]+=1;
				j+=1  
	print "labels: "
	print diff_labels
	print "array: " 
	print cluster_Matrix

	ssn = "labels are: "
	for item in diff_labels:
		ssn = ssn + item + ","

	copy_of_cluster_Matrix = np.copy(cluster_Matrix)
	val = get_max(copy_of_cluster_Matrix)
	summ = np.sum(cluster_Matrix)

	accu = (val*1.0)/(summ*1.0) *100
	ssf = "Correct: " + str(val) + "/" + str(summ) + "\tAccuracy: " + str(accu) + "%"
	np.savetxt("resultsAVG/Cluster_Matrix.csv", cluster_Matrix, fmt='%-4.1d', delimiter=' ', newline='\n', header=ssn, footer='')
	print ssf