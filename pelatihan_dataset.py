# USAGE
# python pelatihan_dataset.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# impor paket yang diperlukan
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# membangun parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# muat embeddings wajah
print("[INFO] memuat hiasan wajah...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# menyandikan label
print("[INFO] label pengodean...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# melatih model yang digunakan untuk menerima embeddings wajah dan 128-d
# kemudian menghasilkan pengenalan wajah yang sebenarnya
print("[INFO] model pelatihan...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()