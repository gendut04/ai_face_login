# USAGE
# python pengenalan_realtime.py --detector programming_bigproject --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle 

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# muat detektor wajah bersambung kami dari disk
print("[INFO] memuat detektor wajah...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# memuat model penyisipan wajah berseri dari serial
print("[INFO] memuat pengenal wajah...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# muat model pengenalan wajah yang sebenarnya bersama dengan label enkoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# inisialisasi aliran video, lalu biarkan sensor kamera dimulai
print("[INFO] mulai streaming video...")
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)

# mulai penaksiran throughput FPS
fps = FPS().start()

# lingkaran bingkai dari aliran file video
while True:
    	# ambil bingkai dari aliran video berulir
	frame = vs.read()
	#(grabbed, frame) = vs.read()
    
	if frame is None:
		break

	# mengubah ukuran frame untuk memiliki lebar 600 piksel (sementara
    # mempertahankan rasio aspek), lalu ambil gambar
    # dimensi
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# membangun blob dari gambar
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# menerapkan pendeteksi wajah berbasis pembelajaran OpenCV yang mendalam untuk melokalisasi
    # wajah pada gambar input
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop atas deteksi
	for i in range(0, detections.shape[2]):
		# ekstrak kepercayaan (mis., probabilitas) yang terkait dengan
        # prediksi
		confidence = detections[0, 0, i, 2]

		# saring deteksi lemah
		if confidence > args["confidence"]:
			# menghitung (x, y) -koordinat dari kotak pembatas untuk
            # muka
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ekstrak ROI wajah
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# Pastikan lebar dan tinggi wajah cukup besar
			if fW < 20 or fH < 20:
				continue

			# buat gumpalan untuk ROI wajah, lalu lewati gumpalan
            # melalui model penyisipan wajah kami untuk mendapatkan 128-d
            # kuantifikasi wajah
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# melakukan klasifikasi untuk mengenali wajah
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# menggambar kotak pembatas wajah bersama dengan
            # kemungkinan terkait
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

			print("Hasil deteksi : ", name)

	# perbarui penghitung FPS
	fps.update()

	# perlihatkan frame output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# jika tombol `q` ditekan, patahkan dari loop
	if key == ord("q"):
		break

# hentikan timer dan tampilkan informasi FPS
fps.stop()
print("[INFO] waktu berlalu: {:.2f}".format(fps.elapsed()))
print("[INFO] jumlah FPS sekitar: {:.2f}".format(fps.fps()))

# lakukan sedikit pembersihan
cv2.destroyAllWindows()
vs.stop()