from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2, time
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import tkinter


from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import * 
from PIL import Image
# import Image
class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])
	msg = 'Recognized:', '"' + recognized[0] + '"'
	msg2 = '   Probability:', probability[0]
	
	messagebox.showinfo("Result",msg +msg2)


def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)

	# infer text on test image
	else:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
		infer(model, FilePaths.fnInfer)

def gui():
        global screen
        screen = tkinter.Tk()

        s = Style()
        s.configure('My.TFrame', background='#5ba8a0')

        mail1 = Frame(screen, style='My.TFrame')
        mail1.place(height=400, width=647,x=0, y=0)
        mail1.config()

        head = tkinter.Label(text="Handwritten text recognition using tensorflow",bg="#5ba8a0",fg="#ffffff",font = "Helvetica 16 bold italic").place(x=70,y=70)

        screen.geometry("647x400")
        screen.title("Handwritten text recognition using tensorflow")
        l = tkinter.Label(text="Captuer image by camera",bg="#5ba8a0",fg="#ffffff",).place(x=70,y=210)
        button = tkinter.Button(text="Open Camera", bg="#7dc9e7",fg="#ffffff",command=camera)
        button.place(x=100,y=250)


        l2 = tkinter.Label(text="Open saved image",bg="#5ba8a0",fg="#ffffff").place(x=393,y=210)
        button2 = tkinter.Button(text="Pick Image", bg="#7dc9e7",fg="#ffffff",command=imagepicker)
        button2.place(x=410,y=250)
        screen.mainloop()

def imagepicker():
        filename = filedialog.askopenfilename(initialdir="/",title = "Select a image", filetype=(("PNG","*.png"),("JPEG","*.jpg"),("All Files","*.*")))
        FilePaths.fnInfer = filename
        main()
        
def camera():
        global video
        video = cv2.VideoCapture(0)
           #make_480p()
        def change_res(width, height):
              video.set(9, width)
              video.set(16, height)
              
        #change_res(32, 32)
        change_res(16, 16)
        a = 0
        while True:
              a = a + 1
              check, frame = video.read()
              print(check)
              print(frame)

              grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              thresh, blackAndWhiteImage = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

              cv2.imshow("Image", blackAndWhiteImage)
              # cv2.waitKey(0)
              key = cv2.waitKey(1)
              if key == ord('q'):
                 break
                
        cv2.imshow("Image", blackAndWhiteImage)
        cv2.destroyAllWindows() 
        video.release()
        cv2.imwrite("../data/test.png", blackAndWhiteImage)
        img = Image.open("../data/test.png")
        img = img.convert('1')
        img.save("../data/test.png")
        FilePaths.fnInfer = '../data/test.png'
        main()

def make_480p():
              video.set(9, 128)
              video.set(16, 32)


if __name__ == '__main__':
	gui()

