# USAGE
# python buat_dataset.py --dataset dataset --embeddings output/embeddings.pickle --detector programming_bigproject --embedding-model openface_nn4.small2.v1.t7

# impor paket yang diperlukan
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# membangun parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# muat detektor wajah bersambung kami dari disk
print("[INFO] memuat bigproject Programing...")
print("[INFO] memuat detektor wajah...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# memuat model penyisipan wajah berseri dari serial
print("[INFO] memuat pengenal wajah...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# ambil jalur ke gambar input dalam dataset kita
print("[INFO] mengukur wajah...")
imagePaths = list(paths.list_images(args["dataset"]))

# inisialisasi daftar embeddings wajah yang diekstraksi dan
# nama orang yang sesuai
knownEmbeddings = []
knownNames = []

# inisialisasi jumlah total wajah yang diproses
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# ekstrak nama orang dari jalur gambar
	print("[INFO] memproses gambar{}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# muat gambar, ubah ukurannya menjadi lebar 600 piksel (sementara
    # mempertahankan rasio aspek), lalu ambil gambar
    # dimensi
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# membangun gumpalan dari gambar
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# menerapkan pendeteksi wajah berbasis pembelajaran OpenCV yang mendalam untuk melokalisasi
    # wajah pada gambar input
	detector.setInput(imageBlob)
	detections = detector.forward()

	# Pastikan setidaknya satu wajah ditemukan
	if len(detections) > 0:
		# kami membuat asumsi bahwa setiap gambar hanya memiliki SATU
    # wajah, jadi temukan kotak pembatas dengan probabilitas terbesar
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# memastikan bahwa deteksi dengan probabilitas terbesar juga
        # berarti uji probabilitas minimum kami (dengan demikian membantu menyaring
        # deteksi lemah)
		if confidence > args["confidence"]:
			# menghitung (x, y) -koordinat dari kotak pembatas untuk
            # muka
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ekstrak ROI wajah dan ambil dimensi ROI
			face = image[startY:endY, startX:endX]
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

			# tambahkan nama orang + wajah yang sesuai
            # menanamkan ke daftar masing-masing
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# buang embeddings wajah + nama ke disk
print("[INFO] membuat serialisasi {} penyandian ...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()