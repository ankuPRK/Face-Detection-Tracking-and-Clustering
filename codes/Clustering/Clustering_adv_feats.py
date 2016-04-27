# This code takes as input the LBP-TOP, Volume-LBP, HOG-TOP or Volume-HOG features
# extracted using the cpp code, and runs clustering on those features. 
# Results are stored in "results/"

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
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
def get_max(Cluster_Matrix):
	print "\nfnn:"
	maxVal = 0
	fl = Cluster_Matrix.flatten()
	print fl
	if Cluster_Matrix.shape[0] == 1 or Cluster_Matrix.shape[1] == 1:
		maxVal = max(fl)
	elif all(fl[x]==fl[0] for x in range(fl.shape[0]) ) :
		maxVal = fl[0] * min([Cluster_Matrix.shape[0], Cluster_Matrix.shape[1]])
	else:
		val = np.amax(fl)
		ls = []
		for i in range(Cluster_Matrix.shape[0]):
			for j in range(Cluster_Matrix.shape[1]):
				if Cluster_Matrix[i,j] == val:
					ls.append([i,j])
		lsmx = []
		for item in ls:
			l = np.delete(Cluster_Matrix,item[0],0)
			l = np.delete(l,item[1],1)
			mx = val + get_max(l)
			lsmx.append(mx)
		maxVal = max(lsmx)
	print "maxVal: " + str(maxVal)
	return maxVal

################################################_main_##################################################
if __name__ == '__main__':

	sampleNo = 25
	numberOfClusters = 4
	labelIspresent = True
	numberOfCharacters = 4

	ls_TrackFeaturesFileList = os.listdir("histodata/")
	ls_TrackFeaturesFileList.sort()
	print len(ls_TrackFeaturesFileList)
	ls_img_for_representing_Track = []

#Read the txt files, each file containing feature for one track
	for filename in ls_TrackFeaturesFileList:
		fn2 = filename.strip(".txt")
		fn3 = "1" + fn2[1:] + ".jpg"	
		#one jpg representative image for each feature txt file
		#Clustering is done on features, and each feature file correspond to one image file,
		#and the program, after doing clustering on features, seperates the corresponding image files
		#into clusters and saves the representative image files, 
		#so that we may see which faces are clustered where.
		ls_img_for_representing_Track.append(fn3)

	ls_FeaturesForTracks = []
	for i in range(len(ls_TrackFeaturesFileList)):
		histS = "histodata/" + ls_TrackFeaturesFileList[i]
		with open(histS) as f:
			content = f.readlines()
		fs = []
		for x in content:
			fs.append(float(x))
		ls_FeaturesForTracks.append(np.array(fs))

	np_FeaturesForTracks = np.array(ls_FeaturesForTracks)
	print "size of np_FeaturesForTracks: " + str(np_FeaturesForTracks.shape)

#Dimension Reduction via Principal Component Analysis
	pcm = PCA(numberOfClustersomponents=11,whiten =True)
	np_FeaturesForTrack_PCAed = pcm.fit_transform(np_FeaturesForTracks)

	#model = cluster.KMeans(numberOfClusterslusters=numberOfClusters, max_iter=300,n_init=100,init="k-means++",)
	model = cluster.SpectralClustering(numberOfClusterslusters=numberOfClusters, eigen_solver=None, random_state=None, n_init=100, gamma=0.0001, affinity='rbf', 
		n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=10, coef0=1, kernel_params=None)

	print "Start fitting into clusters"
	ls_Y_ClusterLabel = model.fit_predict(np_FeaturesForTrack_PCAed)
	print "Done."
	np_Y_ClusterLabel = np.array(ls_Y_ClusterLabel)
	print np_Y_ClusterLabel.shape
	print np_Y_ClusterLabel

	for i in range(0,numberOfClusters):
		newpath = "results/" + str(i) + "/"
		if not os.path.isdir(newpath):
			os.makedirs(newpath)
	while i<np_Y_ClusterLabel.shape[0]:
		grupNo = np_Y_ClusterLabel[i]
		newpath = "results/" + str(grupNo) + "/"
	   	filename = "imgdata/" + ls_img_for_representing_Track[i]
	   	image_array = np.array(Image.open(filename))
		ss = newpath + str(i) + ".jpg"
		im = Image.fromarray(image_array)
		im.save(ss)
		i+=1

##################if label is present then calculate the accuracies#####################################
	if labelIspresent == False:
		exit()

	labels = open("labels.txt","r")
	s = labels.read()
	lb_y = s.split(',')
	del lb_y[-1]

	i=0
	files_list = []

	for i in range(0,numberOfClusters):
		newpath = "results/" + str(i) + "/"
		ss = newpath + str(i) + ".txt"
		fp = open(ss,"w")
		files_list.append(fp)


	i=0
	while i<np_Y_ClusterLabel.shape[0]:
		grupNo = np_Y_ClusterLabel[i]
		ss2 = str(lb_y[i]) + ","
		files_list[grupNo].write(ss2)	
		i+=1	
	print "Done. Making the Cluster-Matrix: "

	for i in range(0,numberOfClusters):
		files_list[i].close()


	diff_labels = []	#the list of unique labels used to label the data, i.e for making the label.txt file
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
	Cluster_Matrix = np.zeros([numberOfClusters,len(diff_labels)], dtype=int)

	i=0
	for i in range(0,numberOfClusters):
		ssn = "results/" + str(i) + "/" + str(i) + ".txt"
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
					Cluster_Matrix[i][j]+=1;
				j+=1  
	print "labels: "
	print diff_labels
	print "array: " 
	print Cluster_Matrix

	ssn = "labels are: "
	for item in diff_labels:
		ssn = ssn + item + ","

	copy_of_Cluster_Matrix = np.copy(Cluster_Matrix)
	val = get_max(copy_of_Cluster_Matrix)
	Total = np.sum(Cluster_Matrix)

	accu = (val*1.0)/(Total*1.0) *100
	ssf = "Correct: " + str(val) + "/" + str(Total) + "\tAccuracy: " + str(accu) + "%"

	np.savetxt("results/clusterMatrix.csv", Cluster_Matrix, fmt='%-4.1d', delimiter=' ', newline='\n', header=ssn, footer=ssf)
	print ssf