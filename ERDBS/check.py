import os

run = 0
dirs = []

def isBetter(psnr, seq):
    global run
    global dirs
    models = dirs
    if run == 0:
        models = os.listdir('parameters')
        models = [m.split('-') for m in models]
        models = [[float(m[0]), m[1][:-4]] for m in models]
        dirs = models
    dirs.append([psnr, seq])
    run += 1
    for m in models:
        if m[0] > psnr and len(m[1]) == len(seq):
            return False

    return True