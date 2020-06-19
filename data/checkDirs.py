import os.path


checkDirs = ['sentences/', 'sentences/a01/a01-000u/']
checkFiles = ['sentences.txt', 'test.png', 'sentences/a01/a01-000u/a01-000u-s00-00.png']


for f in checkDirs:
	if os.path.isdir(f):
		print('[OK]', f)
	else:
		print('[ERR]', f)


for f in checkFiles:
	if os.path.isfile(f):
		print('[OK]', f)
	else:
		print('[ERR]', f)
