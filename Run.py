from flask import Flask, render_template, request, redirect, url_for
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib import config
import matplotlib.pyplot as plt
from statistics import mean
import time, schedule, csv
import argparse, imutils
import time, dlib, cv2, datetime
import numpy as np
import threading

def run():
	try: 
		# construct the argument parse and parse the arguments
		testing = []
		time_in = []
		time_out = []
		avg_times = []
		avg_seconds = []
		tested=[]
		num_people = []
  		#initialise all inidices of tested with False
		for i in range(0,10000):
			tested.append(0)
			time_in.append('0')
			time_out.append('0')
		# initialize the video stream and allow the cammera sensor to warmup		
		ap = argparse.ArgumentParser()
		ap.add_argument("-p", "--prototxt", required=False,
			help="path to Caffe 'deploy' prototxt file")
		ap.add_argument("-m", "--model", required=True,
			help="path to Caffe pre-trained model")
		ap.add_argument("-i", "--input", type=str,
			help="path to optional input video file")
		ap.add_argument("-i2", "--input2", type=str,
			help="path to optional input video file")
		ap.add_argument("-o", "--output", type=str,
			help="path to optional output video file")
		ap.add_argument("-c", "--confidence", type=float, default=0.4,
			help="minimum probability to filter weak detections")
		ap.add_argument("-s", "--skip-frames", type=int, default=30,
			help="# of skip frames between detections")
		args = vars(ap.parse_args())

		# initialize the list of class labels MobileNet SSD was trained to detect
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		# load our serialized model from disk
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

		name = threading.currentThread().getName()
		# if a video path was not supplied, grab a reference to the ip camera
		if not args.get("input", False):
			print("[INFO] Starting the stream..")
			vs = cv2.VideoCapture(name)
			if vs.isOpened():
				rval, frame = vs.read()

		# otherwise, grab a reference to the video file
		else:
			print("[INFO] Starting the video..")
			vs = VideoStream(name).start()

		# initialize the video writer 
		writer = None

		# initialize the frame dimensions 
		W = None
		H = None

		ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # instantiate the centroid tracker
		trackers = [] # instantiate the list of trackers
		trackableObjects = {} # instantiate the dictionary of trackable objects
		totalFrames = 0 # initialize the total number of frames processed

		# start the frames per second throughput estimator
		fps = FPS().start()


		# loop over frames from the video stream
		while rval:
			# next frame
			rval, frame = vs.read()

			# end of the video
			if args["input"] is not None and frame is None:
				break

			# resize frame and convert to RGB
			frame = imutils.resize(frame, width = 500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]
			# if we should write video, initialize the writer
			if args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(W, H), True)
			rects = []

			# when no person is detected, we will use the object detector
			if totalFrames % args["skip_frames"] == 0:
				trackers = []
				# convert fram to blob and obtain detections
				blob = cv2.dnn.blobFromImage(frame, 0.006000, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					confidence = detections[0, 0, i, 2] #check the confidence of the detection
					if confidence > args["confidence"]: 
						# get the index of the class label from the detections
						idx = int(detections[0, 0, i, 1])
						if CLASSES[idx] != "person": #if its not a person, ignore it
							continue
						# find the co-ordinates of the bounding box for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")
						# construct dlib rectangle object from the co-ordinates and then update the tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)
						# add the tracker to our list of trackers 
						trackers.append(tracker)
				
			# if person is detected, we will use the tracker
			else:
				
				# loop over the trackers
				for tracker in trackers:
					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()
					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())
					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))


			objects = ct.update(rects)
   			
			# loop over the tracked objects
			#delete all entries of testing
			testing.clear()
			for (objectID, centroid) in objects.items():
				
				testing.append(objectID)
				# check to see if a trackable object exists for the current object ID
				if tested[objectID] == 0:
					t = time.localtime()
					#timer1[object] = t
					current_time = time.strftime("%H:%M:%S", t)
					time_in[objectID]=current_time
					tested[objectID]=1
					print("Person ",objectID,"came in at",time_in[objectID])
				# check to see if objectID is in the list of trackers
				to = trackableObjects.get(objectID, None)
				# if there is no existing trackable object, create one

				if to is None:
					to = TrackableObject(objectID, centroid)
				# but if it exists
				else:
					# check in which direction the object is moving, then
					y = [c[1] for c in to.centroids]
					to.centroids.append(centroid)

				trackableObjects[objectID] = to			
					
				# display id and centroid on frame
				text = "{}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
    
			for i in range(10000):
				if tested[i]==1:
					# if is not there in testing
					if i not in testing:
						tested[i]=2
						# calculating after how much time they left
						t = time.localtime()
						current_time = time.strftime("%H:%M:%S", t)
						time_out[i]=current_time
						ls1=time_out[i].split(":")
						ls1 = [int(x) for x in ls1]
						ls2=time_in[i].split(":")
						ls2 = [int(x) for x in ls2]
						seconds = (ls1[0]*3600)+(ls1[1]*60)+ls1[2]-(ls2[0]*3600)-(ls2[1]*60)-ls2[2]
						print("Person ",i,"went out at",time_out[i],"after",seconds,"seconds")
						#create a file name.txt where name in variable name
						name_test = str(name).replace("/","")
						name_test = str(name_test).replace(":","")
						name_test = str(name_test).replace("@","")
						file = open(str(name_test)+".txt","w")
						if seconds >= config.threshold_time:
							avg_times.append(seconds)
							avg_seconds.append(mean(avg_times))
							num_people.append(len(num_people)+1)					
							#write num_people and avg_seconds to the file
							file.write(str(num_people)+"\n")
							file.write(str(avg_seconds)+"\n")
							file.close()

			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame without minimising the window whith name of current thread
			cv2.imshow(name, frame)
			#dont minimise frame
			#cv2.setWindowProperty("CodonSoft People Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

			#fps updater
			totalFrames += 1
			fps.update()

		# display time and fps information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	except:
		#do nothing
		pass

def runner(n):
	try: 
		# construct the argument parse and parse the arguments
		testing = []
		time_in = []
		time_out = []
		avg_times = []
		avg_seconds = []
		tested=[]
		num_people = []
  		#initialise all inidices of tested with False
		for i in range(0,10000):
			tested.append(0)
			time_in.append('0')
			time_out.append('0')
		# initialize the video stream and allow the cammera sensor to warmup		
		ap = argparse.ArgumentParser()
		ap.add_argument("-p", "--prototxt", required=False,
			help="path to Caffe 'deploy' prototxt file")
		ap.add_argument("-m", "--model", required=True,
			help="path to Caffe pre-trained model")
		ap.add_argument("-i", "--input", type=str,
			help="path to optional input video file")
		ap.add_argument("-i2", "--input2", type=str,
			help="path to optional input video file")
		ap.add_argument("-o", "--output", type=str,
			help="path to optional output video file")
		ap.add_argument("-c", "--confidence", type=float, default=0.4,
			help="minimum probability to filter weak detections")
		ap.add_argument("-s", "--skip-frames", type=int, default=30,
			help="# of skip frames between detections")
		args = vars(ap.parse_args())

		# initialize the list of class labels MobileNet SSD was trained to detect
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		# load our serialized model from disk
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

		name = threading.currentThread().getName()
		# if a video path was not supplied, grab a reference to the ip camera
		if not args.get("input", False):
			print("[INFO] Starting the stream..")
			print(n)
			vs = cv2.VideoCapture(n)
			if vs.isOpened():
				rval, frame = vs.read()

		# otherwise, grab a reference to the video file
		else:
			print("[INFO] Starting the video..")
			vs = VideoStream(name).start()

		# initialize the video writer 
		writer = None

		# initialize the frame dimensions 
		W = None
		H = None

		ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # instantiate the centroid tracker
		trackers = [] # instantiate the list of trackers
		trackableObjects = {} # instantiate the dictionary of trackable objects
		totalFrames = 0 # initialize the total number of frames processed

		# start the frames per second throughput estimator
		fps = FPS().start()


		# loop over frames from the video stream
		while rval:
			# next frame
			rval, frame = vs.read()
   
			# end of the video
			if args["input"] is not None and frame is None:
				break

			# resize frame and convert to RGB
			frame = imutils.resize(frame, width = 500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]
			# if we should write video, initialize the writer
			if args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(W, H), True)
			rects = []

			# when no person is detected, we will use the object detector
			if totalFrames % args["skip_frames"] == 0:
				trackers = []
				# convert fram to blob and obtain detections
				blob = cv2.dnn.blobFromImage(frame, 0.007000, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					confidence = detections[0, 0, i, 2] #check the confidence of the detection
					if confidence > args["confidence"]: 
						# get the index of the class label from the detections
						idx = int(detections[0, 0, i, 1])
						if CLASSES[idx] != "person": #if its not a person, ignore it
							continue
						# find the co-ordinates of the bounding box for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")
						# construct dlib rectangle object from the co-ordinates and then update the tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)
						# add the tracker to our list of trackers 
						trackers.append(tracker)
				
			# if person is detected, we will use the tracker
			else:
				
				# loop over the trackers
				for tracker in trackers:
					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()
					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())
					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))


			objects = ct.update(rects)
   			
			# loop over the tracked objects
			#delete all entries of testing
			testing.clear()
			for (objectID, centroid) in objects.items():
				
				testing.append(objectID)
				# check to see if a trackable object exists for the current object ID
				if tested[objectID] == 0:
					t = time.localtime()
					#timer1[object] = t
					current_time = time.strftime("%H:%M:%S", t)
					time_in[objectID]=current_time
					tested[objectID]=1
					print("Person ",objectID,"came in at",time_in[objectID])
				# check to see if objectID is in the list of trackers
				to = trackableObjects.get(objectID, None)
				# if there is no existing trackable object, create one

				if to is None:
					to = TrackableObject(objectID, centroid)
				# but if it exists
				else:
					# check in which direction the object is moving, then
					y = [c[1] for c in to.centroids]
					to.centroids.append(centroid)

				trackableObjects[objectID] = to			
					
				# display id and centroid on frame
				text = "{}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
    
			for i in range(10000):
				if tested[i]==1:
					# if is not there in testing
					if i not in testing:
						tested[i]=2
						t = time.localtime()
						current_time = time.strftime("%H:%M:%S", t)
						time_out[i]=current_time
						ls1=time_out[i].split(":")
						ls1 = [int(x) for x in ls1]
						ls2=time_in[i].split(":")
						ls2 = [int(x) for x in ls2]
						seconds = (ls1[0]*3600)+(ls1[1]*60)+ls1[2]-(ls2[0]*3600)-(ls2[1]*60)-ls2[2]
						print("Person ",i,"went out at",time_out[i],"after",seconds,"seconds")
						#create a file name.txt where name is variable name
						n_test = str(n).replace("/","")
						n_test = str(n_test).replace(":","")
						n_test = str(n_test).replace("@","")
						file = open(str(n_test)+".txt","w")
						if seconds >= config.threshold_time:
							avg_times.append(seconds)
							avg_seconds.append(mean(avg_times))
							num_people.append(len(num_people)+1)					
							#write num_people and avg_seconds to the file
							file.write(str(num_people)+"\n")
							file.write(str(avg_seconds)+"\n")
							file.close()

						
			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame without minimising the window whith name of current thread
			cv2.imshow(name, frame)
			#dont minimise frame
			#cv2.setWindowProperty("CodonSoft People Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

			#fps updater
			totalFrames += 1
			fps.update()

		# display time and fps information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	except:
		#do nothing
		pass

if config.Scheduler:
	#schedule whenever u want to run the code
	schedule.every().day.at("09:00").do(run)

	while 1:
		schedule.run_pending()

else:
    #x equal to length of config.url
    x = config.url.__len__()
    for i in config.url:
        #t = threading.Thread(target=run, name = i)
        if type(i) is str:
            i_temp = str(i).replace("/","")
            i_temp = str(i_temp).replace(":","")
            i_temp = str(i_temp).replace("@","")
            #create file of name i_temp.txt
            with open(str(i_temp)+".txt", "w") as f:
                pass	
            t = threading.Thread(target=run, name = i)
        else:
            t = threading.Thread(target=runner, args = (i,))
        t.start()
    for thread in threading.enumerate():
        if thread.name != "MainThread":
            thread.join()
    for i in config.url:
        # open file i.txt where i is the variable
        i_temp = str(i).replace("/","")
        i_temp = str(i_temp).replace(":","")
        i_temp = str(i_temp).replace("@","")
        file = open(str(i_temp)+".txt","r")
        # read fthe contents of the files
        first_line = file.readlines()
        x = len(first_line)
        first_l = []
        second_l = []
        print(x)
        if x > 0:
            first_line[0] = first_line[0].replace("\n","")
            first_line[0] = first_line[0].replace("[","")
            first_line[0] = first_line[0].replace("]","")
            first_line[0] = first_line[0].replace(" ","")
            first_line[1] = first_line[1].replace("\n","")
            first_line[1] = first_line[1].replace("[","")
            first_line[1] = first_line[1].replace("]","")
            first_line[1] = first_line[1].replace(" ","")
            first_l = list(first_line[0].split(","))
            second_l = list(first_line[1].split(","))
            first_l = [eval(x) for x in first_l]
            second_l = [eval(x) for x in second_l]
        # plot the graph
        plt.plot(first_l,second_l,label=str(i))
        plt.xlabel('Number of people')
        plt.ylabel('Average time')
        plt.legend()							
    plt.show()

        
        
        